"""Verify E03 initialization, weighting placement and frozen-factor gradients."""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import build_model, configure, initialization_digest, now
from src.data import OilSpillTiles, shared
from src.runtime import apply_runtime


def main(run):
    plan = json.loads((run/'protocol.json').read_text())
    runtime = json.loads((run/'runtime.json').read_text())
    variant = 'raw_input'
    reference_run = Path(plan['weighting_reference_run'])
    reference_plan = json.loads((reference_run/'protocol.json').read_text())
    historical = json.loads((reference_run/'models/reliability_detail/initialization.json').read_text())
    data = OilSpillTiles(plan['cache_root'], 'tr', plan['band_indices'])
    images = torch.stack([data[i][0] for i in range(2)])
    targets = torch.stack([data[i][1] for i in range(2)])
    configure(plan['seed'])
    reference = build_model(reference_plan, 'reliability_detail').eval()
    expected = initialization_digest(reference)
    assert all(historical[k] == v for k, v in expected.items())
    reference_parameters = reference.parameter_report()
    apply_runtime(reference, runtime)
    with torch.no_grad():
        reference_logits = reference(images)
    del reference
    gc.collect()
    configure(plan['seed'])
    model = build_model(plan, variant).eval()
    initial = initialization_digest(model)
    assert initial == expected, 'Shared tensors or CPU RNG changed'
    assert model.parameter_report() == reference_parameters
    apply_runtime(model, runtime)
    with torch.no_grad():
        output = model(images)
    torch.testing.assert_close(output, reference_logits, rtol=0, atol=0)
    checks = dict(historical_state_and_rng_identical=True, parameter_counts_identical=True,
                  unit_weight_initial_logits_identical=True)
    del output, reference_logits
    model.cuda()
    images, targets = images.cuda(), targets.cuda()
    embedding = model.network.encoder.patch_embed
    captured = {}

    def content_hook(module, args):
        captured['content_input'] = args[0].detach().clone()

    def factors_hook(module, args, output):
        assert all(not t.requires_grad for t in output)
        captured['factors'] = output

    h1 = embedding.content_feature_extractor.register_forward_pre_hook(content_hook)
    h2 = embedding.matrix_generator.register_forward_hook(factors_hook)
    weights = torch.linspace(.5, 1.5, images.shape[1], device='cuda').repeat(2, 1).requires_grad_()
    tokens = model.tokenize(images, weights)
    torch.testing.assert_close(captured['content_input'], images * weights[:, :, None, None], rtol=0, atol=0)
    u, v, bias = captured['factors']
    ps = embedding.kernel_size
    patches = F.unfold(images * weights[:, :, None, None], ps, stride=ps).transpose(1, 2)
    patches = patches.reshape(2, -1, images.shape[1], ps * ps)
    manual = torch.einsum('bncr,bcer->bne', torch.einsum('bncp,bcrp->bncr', patches, v), u)
    manual = (manual + bias[:, None, :]) * embedding.scaler
    torch.testing.assert_close(tokens, manual, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(tokens.square().mean(), weights)[0]
    expected_gradient = torch.autograd.grad(manual.square().mean(), weights)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().sum() > 0
    model.weighting_location = 'aggregation'
    with torch.no_grad():
        aggregation_tokens = model.tokenize(images, weights)
    torch.testing.assert_close(captured['content_input'], images, rtol=0, atol=0)
    assert (tokens.detach() - aggregation_tokens).abs().max() > 1e-5
    model.weighting_location = 'raw_input'
    h1.remove(); h2.remove()
    checks.update(raw_factors_receive_weighted_input=True, reference_factors_receive_original_input=True,
                  weights_applied_once=True, detached_factor_gradient_formula_matches=True,
                  nonuniform_paths_differ=True)
    del tokens, manual, patches, captured, aggregation_tokens, u, v, bias

    def detail_hook(module, args):
        torch.testing.assert_close(args[0], images, rtol=0, atol=0)
        assert not args[1].requires_grad
        assert args[2].requires_grad

    handle = model.spectral_detail.register_forward_pre_hook(detail_hook)
    # Nonuniform reliability and nonzero detail logits expose both paths.
    with torch.no_grad():
        model.reliability.score.weight.fill_(.15)
        model.spectral_detail.output.weight.fill_(.01)
    output = model(images)
    F.cross_entropy(output, targets, ignore_index=255).backward()
    gradient = model.reliability.score.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.network.encoder.parameters())
    assert model.spectral_detail.output.weight.grad.abs().sum() > 0
    handle.remove()
    checks.update(reliability_receives_segmentation_gradient=True, encoder_frozen=True,
                  detail_receives_original_input_and_detached_weights=True,
                  decoder_semantic_context_gradient_retained=True)
    shared.save_json(run/'weighting_preflight.json', dict(
        status='passed', checked_at=now(), seed=plan['seed'], initialization=initial,
        reference_run=str(reference_run), checks=checks, parameters=reference_parameters))
    print('PASS weighting-location preflight seed', plan['seed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    main(parser.parse_args().run_dir.resolve())
