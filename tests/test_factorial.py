"""Check matched initialization and actual factor removal on real training tiles."""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import build_model, configure, initialization_digest, now
from src.data import OilSpillTiles, shared
from src.runtime import apply_runtime


def main(run_dir):
    plan = json.loads((run_dir/'protocol.json').read_text())
    runtime = json.loads((run_dir/'runtime.json').read_text())
    data = OilSpillTiles(plan['cache_root'], 'tr', plan['band_indices'])
    images = torch.stack([data[i][0] for i in range(2)])
    reference, initial = None, None
    records = {}
    for variant, flags in plan['variants'].items():
        configure(plan['seed'])
        model = build_model(plan, variant).eval()
        state = initialization_digest(model)
        if initial is None:
            initial = state
        assert state == initial, (variant, 'initial weights/RNG mismatch')
        apply_runtime(model, runtime)
        groups = model.parameter_report()['groups']
        assert bool(groups['reliability']) == flags['use_reliability']
        assert bool(groups['spectral_detail']) == flags['spectral_detail']
        assert all(groups[k] == 0 for k in ['condition', 'routers', 'adapters',
                                          'local_reliability', 'spatial_routers', 'spatial_adapters'])
        calls = []

        def guard_reliability(module, args):
            calls.append('reliability')
            assert flags['use_reliability'], 'Disabled reliability estimator was evaluated'

        def guard_detail(module, args):
            calls.append('detail')
            assert flags['spectral_detail']
            if not flags['use_reliability']:
                torch.testing.assert_close(args[1], torch.ones_like(args[1]), rtol=0, atol=0)

        model.reliability.register_forward_pre_hook(guard_reliability)
        model.spectral_detail.register_forward_pre_hook(guard_detail)
        with torch.no_grad():
            output = model(images)
        assert torch.isfinite(output).all()
        assert ('reliability' in calls) == flags['use_reliability']
        assert ('detail' in calls) == flags['spectral_detail']
        torch.testing.assert_close(model.last_diagnostics['weights'],
                                   torch.ones_like(model.last_diagnostics['weights']), rtol=0, atol=0)
        if reference is None:
            reference = output.clone()
        torch.testing.assert_close(output, reference, rtol=0, atol=0)

        # For B, train the real detail branch with the estimator forbidden.
        # Its zero-initialized output learns on step one; step two must carry
        # gradient into projection, high-frequency filter and semantic gate.
        if variant == plan['factorial_arms']['B']:
            optimizer = torch.optim.SGD(model.spectral_detail.parameters(), lr=.01)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                output = model(images)
                output[:, 1].square().mean().backward()
                for parameter in model.reliability.parameters():
                    assert parameter.grad is None
                assert all(p.grad is None for p in model.network.encoder.parameters())
                optimizer.step()
            for module in [model.spectral_detail.projection, model.spectral_detail.filter,
                           model.spectral_detail.gate, model.spectral_detail.output]:
                gradient = module.weight.grad
                assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            torch.testing.assert_close(model.last_diagnostics['weights'],
                                       torch.ones_like(model.last_diagnostics['weights']), rtol=0, atol=0)
            del optimizer
        records[variant] = dict(initialization=state, trainable_groups=groups,
                                initial_predictions_match=True, disabled_paths_bypassed=True)
        print('PASS '+variant, flush=True)
        del model, output
        gc.collect()
    shared.save_json(run_dir/'factorial_preflight.json', dict(status='passed',
        checked_at=now(), device='cpu', initialization=initial, variants=records,
        checks=['identical full initial state and CPU RNG', 'identical initial logits',
                'disabled estimator never called', 'unit weights at tokenizer and detail branch',
                'B detail projection/filter/gate gradients with frozen encoder and reliability'],
        note='CPU preflight only; independent three-update CUDA smoke and complete validation precede every formal arm.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    main(parser.parse_args().run_dir.resolve())
