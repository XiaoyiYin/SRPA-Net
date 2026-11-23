"""Audit matched initialization and both gradient routes in the location control."""
import argparse
import gc
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train import build_model,configure,initialization_digest,now
from src.data import OilSpillTiles,shared
from src.runtime import apply_runtime


def main(run):
    plan=json.loads((run/'protocol.json').read_text());runtime=json.loads((run/'runtime.json').read_text())
    reference_run=Path(plan['weighting_reference_run'])
    reference_plan=json.loads((reference_run/'protocol.json').read_text())
    historical=json.loads((reference_run/'models/reliability_detail/initialization.json').read_text())
    dataset=OilSpillTiles(plan['cache_root'],'tr',plan['band_indices'])
    images=torch.stack([dataset[i][0] for i in range(2)])
    targets=torch.stack([dataset[i][1] for i in range(2)])
    configure(plan['seed']);reference=build_model(reference_plan,'reliability_detail').eval()
    initial=initialization_digest(reference)
    assert all(historical[k]==v for k,v in initial.items())
    parameters=reference.parameter_report()
    trainable={name:p.requires_grad for name,p in reference.named_parameters()}
    apply_runtime(reference,runtime)
    with torch.no_grad():expected_logits=reference(images)
    del reference;gc.collect()
    for variant in ['aggregation','raw_input']:
        configure(plan['seed']);model=build_model(plan,variant).eval()
        assert initialization_digest(model)==initial
        assert model.parameter_report()==parameters
        assert {name:p.requires_grad for name,p in model.named_parameters()}==trainable
        assert model.detail_use_reliability and not model.detail_detach_weights and not model.detail_detach_context
        apply_runtime(model,runtime)
        with torch.no_grad():actual=model(images)
        torch.testing.assert_close(actual,expected_logits,rtol=0,atol=0)
        print('PASS initial tensors/RNG/parameter counts/logits: '+variant,flush=True)
        del actual
        if variant=='aggregation':del model;gc.collect()
    model.cuda();images=images.cuda();targets=targets.cuda()
    embedding=model.network.encoder.patch_embed;captured={}
    def input_hook(module,args):captured['input']=args[0]
    def factor_hook(module,args,output):captured['factors']=output
    h1=embedding.content_feature_extractor.register_forward_pre_hook(input_hook)
    h2=embedding.matrix_generator.register_forward_hook(factor_hook)
    weights=torch.linspace(.5,1.5,images.shape[1],device='cuda').repeat(2,1).requires_grad_()
    tokens=model.tokenize(images,weights)
    torch.testing.assert_close(captured['input'],images*weights[:,:,None,None],rtol=0,atol=0)
    u,v,bias=captured['factors'];assert all(t.requires_grad for t in (u,v,bias))
    ps=embedding.kernel_size
    patches=F.unfold(images*weights[:,:,None,None],ps,stride=ps).transpose(1,2).reshape(2,-1,images.shape[1],ps*ps)
    def formula(uu,vv,bb):
        return (torch.einsum('bncr,bcer->bne',torch.einsum('bncp,bcrp->bncr',patches,vv),uu)+bb[:,None,:])*embedding.scaler
    manual=formula(u,v,bias)
    torch.testing.assert_close(tokens,manual,rtol=0,atol=0)
    g=torch.autograd.grad(tokens.square().mean(),weights,retain_graph=True)[0]
    gm=torch.autograd.grad(manual.square().mean(),weights,retain_graph=True)[0]
    torch.testing.assert_close(g,gm,rtol=1e-6,atol=1e-6)
    patch_only=torch.autograd.grad(formula(u.detach(),v.detach(),bias.detach()).square().mean(),weights,retain_graph=True)[0]
    factor_contribution=float((g-patch_only).abs().sum())
    assert torch.isfinite(g).all() and factor_contribution>1e-7
    model.weighting_location='aggregation'
    aggregate=model.tokenize(images,weights)
    torch.testing.assert_close(captured['input'],images,rtol=0,atol=0)
    assert all(not tensor.requires_grad for tensor in captured['factors'])
    assert float((tokens.detach()-aggregate.detach()).abs().max())>1e-5
    h1.remove();h2.remove()
    del tokens,manual,patches,u,v,bias,aggregate,captured,weights,g,gm,patch_only
    model.weighting_location='raw_input';gc.collect()
    with torch.no_grad():
        model.reliability.score.weight.fill_(.15)
        model.spectral_detail.output.weight.fill_(.01)
    def detail_hook(module,args):
        torch.testing.assert_close(args[0],images,rtol=0,atol=0)
        assert args[1].requires_grad and args[2].requires_grad
    h=model.spectral_detail.register_forward_pre_hook(detail_hook)
    F.cross_entropy(model(images),targets,ignore_index=255).backward()
    assert model.reliability.score.weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in model.network.encoder.parameters())
    h.remove()
    # Isolate the direct detail-to-weight route from the decoder-mediated route.
    weights=torch.ones(2,images.shape[1],device='cuda',requires_grad=True)
    context=torch.zeros(2,64,56,56,device='cuda')
    detail=model.spectral_detail(images,weights,context)
    direct=torch.autograd.grad(detail.square().mean(),weights)[0]
    assert torch.isfinite(direct).all() and direct.abs().sum()>0
    shared.save_json(run/'weighting_preflight.json',dict(status='passed',checked_at=now(),seed=plan['seed'],
        initialization=initial,reference_run=str(reference_run),parameters=parameters,
        checks=dict(both_arms_match_historical_initial_state_and_rng=True,initial_predictions_identical=True,
            same_trainable_parameter_set=True,weights_applied_once=True,
            raw_factor_generation_input_gradient_preserved=True,factor_gradient_contribution_l1=factor_contribution,
            aggregation_factors_independent_of_weights=True,detail_and_context_shared_gradients=True,
            direct_detail_to_weight_gradient=True,encoder_parameters_frozen=True)))
    print('PASS shared-gradient location/formula/factor-gradient/detail-gradient checks',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    main(parser.parse_args().run_dir.resolve())
