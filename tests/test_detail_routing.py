"""Verify matched initialization and the actual two-branch weight routing."""
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
    # Include all three alternatives even when this run trains only one.
    flags = dict(use_reliability=True, use_adapters=False,
                 local_reliability=False, spatial_adapter=False, spectral_detail=True)
    check_plan = dict(plan, variants={
        'detail_only': dict(flags, use_reliability=False),
        'reliability_detail': flags,
        'reliability_detail_raw': dict(flags, detail_use_reliability=False),
    })
    initial, reference = None, None
    records = {}
    for variant in check_plan['variants']:
        configure(plan['seed'])
        model = build_model(check_plan, variant).eval()
        state = initialization_digest(model)
        if initial is None:
            initial = state
        assert state == initial, (variant, 'initial weights/RNG mismatch')
        apply_runtime(model, runtime)
        captured = {}
        original_tokenize = model.tokenize

        def capture_tokenize(x, weights):
            captured['token'] = weights.detach().clone()
            return original_tokenize(x, weights)

        def capture_detail(module, inputs):
            captured['detail'] = inputs[1].detach().clone()

        model.tokenize = capture_tokenize
        model.spectral_detail.register_forward_pre_hook(capture_detail)
        with torch.no_grad():
            output = model(images)
        if reference is None:
            reference = output.clone()
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        # Uniform initialization alone cannot reveal erroneous weight routing.
        # Force nonuniform weights and a nonzero residual for this check only.
        with torch.no_grad():
            model.reliability.score.weight.fill_(0.15)
            model.spectral_detail.output.weight.fill_(0.01)
            output = model(images)
        assert torch.isfinite(output).all()
        if variant == 'detail_only':
            torch.testing.assert_close(captured['token'], torch.ones_like(captured['token']), rtol=0, atol=0)
        else:
            assert (captured['token']-1).abs().max() > 1e-5
        if variant == 'reliability_detail':
            torch.testing.assert_close(captured['detail'], captured['token'], rtol=0, atol=0)
        else:
            torch.testing.assert_close(captured['detail'], torch.ones_like(captured['detail']), rtol=0, atol=0)
        # Remove the gate's weight dependence to isolate the DIRECT detail path.
        # The raw variant intentionally retains the decoder-mediated path.
        captured_decoded = torch.randn(2, 64, 56, 56)
        weights = captured['token'].clone().requires_grad_()
        direct_weights = weights if model.detail_use_reliability else torch.ones_like(weights)
        residual = model.spectral_detail(images, direct_weights, captured_decoded)
        residual.square().mean().backward()
        if variant == 'reliability_detail_raw':
            assert weights.grad is None
        else:
            assert weights.grad is not None and weights.grad.abs().sum() > 0
        for module in [model.spectral_detail.projection, model.spectral_detail.gate]:
            assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
            assert module.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.network.encoder.parameters())
        records[variant] = dict(initialization=state, initial_logits_match=True,
                               nonuniform_weight_routing_checked=True,
                               direct_detail_gradient_checked=True)
        print('PASS '+variant, flush=True)
        del model, output, residual, original_tokenize
        gc.collect()
    if plan['seed'] == 42 and plan.get('reference_run'):
        previous = json.loads((Path(plan['reference_run'])/'factorial_preflight.json').read_text())
        assert initial == previous['initialization'], 'Historical seed42 initialization changed'
    shared.save_json(run_dir/'factorial_preflight.json', dict(
        status='passed', checked_at=now(), initialization=initial, variants=records,
        note='CPU routing checks; independent CUDA smoke/full validation precede each training run.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    main(parser.parse_args().run_dir.resolve())
