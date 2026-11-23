"""Check ablation semantics, gradients, historical initialization and routing."""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mechanism_controls import StaticBandWeights, ControlledSpectralDetail
from src.innovations import SpectralDetail
from src.data import OilSpillTiles, shared
from src.runtime import apply_runtime
from train import build_model, configure, initialization_digest, now


def unit_checks():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    before=torch.get_rng_state().clone()
    static=StaticBandWeights(8)
    assert torch.equal(before,torch.get_rng_state())
    x=torch.randn(2,8,12,12)
    w,quality,_=static(x)
    assert quality is None and torch.equal(w,torch.ones_like(w))
    with torch.no_grad():static.score['logits'].copy_(torch.linspace(-.4,.4,8))
    w,_,_=static(x)
    other,_,_=static(x.flip(0)*7)
    torch.testing.assert_close(w,other,rtol=0,atol=0)
    torch.testing.assert_close(w[0],w[1],rtol=0,atol=0)
    torch.testing.assert_close(w.sum(-1),torch.full((2,),8.),rtol=1e-6,atol=1e-6)
    (w*torch.arange(8.)).sum().backward()
    assert torch.isfinite(static.score['logits'].grad).all() and static.score['logits'].grad.abs().sum()>0
    old=static.score['logits'].detach().clone()
    torch.optim.SGD(static.parameters(),lr=.01).step()
    assert not torch.equal(old,static.score['logits'])
    expected_static=static.state_dict()
    restored=StaticBandWeights(8);restored.load_state_dict(expected_static,strict=True)
    torch.testing.assert_close(static(x)[0],restored(x)[0],rtol=0,atol=0)
    checks={'static_shared_across_inputs_and_batch':True,'static_learnable_and_normalized':True,'static_checkpoint_roundtrip':True}
    for gate,hf in [(True,True),(False,True),(True,False)]:
        torch.manual_seed(7)
        original=SpectralDetail(8,decoder_dim=4,width=8)
        with torch.no_grad():original.output.weight.normal_(0,.02)
        control=ControlledSpectralDetail(8,decoder_dim=4,width=8)
        control.load_state_dict(original.state_dict())
        control.use_semantic_gate=gate;control.use_high_frequency=hf
        control.gate.requires_grad_(gate);control.filter.requires_grad_(hf)
        decoded=torch.randn(2,4,6,6,requires_grad=True)
        weighted=torch.ones(2,8,requires_grad=True)
        projected=F.gelu(original.norm(original.projection(x*weighted[:,:,None,None])))
        manual=projected
        if hf:
            smooth=F.avg_pool2d(F.pad(projected,(1,1,1,1),mode='reflect'),3,1)
            manual=manual+F.gelu(original.filter(projected-smooth))
        if gate:
            context=F.interpolate(original.gate(decoded),size=x.shape[-2:],mode='bilinear',align_corners=False)
            manual=manual*torch.sigmoid(context)
        manual=original.output(manual)
        def disabled(*args,**kwargs):raise AssertionError('Disabled module executed')
        if not gate:control.gate.forward=disabled
        if not hf:control.filter.forward=disabled
        output=control(x,weighted,decoded)
        torch.testing.assert_close(output,manual,rtol=0,atol=0)
        if gate and hf:torch.testing.assert_close(output,original(x,weighted,decoded),rtol=0,atol=0)
        output.square().mean().backward()
        for module,enabled in [(control.projection,True),(control.output,True),(control.filter,hf),(control.gate,gate)]:
            g=module.weight.grad
            if enabled:assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
            else:assert g is None
        assert weighted.grad is not None and weighted.grad.abs().sum()>0
        assert (decoded.grad is not None)==gate
        checks[f'gate_{gate}_high_frequency_{hf}_formula_and_gradients']=True
    print('PASS unit controls',flush=True)
    return checks


def main(run_dir):
    checks=unit_checks()
    plan=json.loads((run_dir/'protocol.json').read_text())
    runtime=json.loads((run_dir/'runtime.json').read_text())
    baseflags=dict(use_reliability=True,use_adapters=False,local_reliability=False,spatial_adapter=False,spectral_detail=True)
    reference_plan=dict(plan,model_family='srpa_innovations_v2',variants={'reference':baseflags})
    data=OilSpillTiles(plan['cache_root'],'tr',plan['band_indices'])
    images=torch.stack([data[i][0] for i in range(2)])
    configure(plan['seed'])
    reference=build_model(reference_plan,'reference').eval()
    reference_initial=initialization_digest(reference)
    historical=json.loads((Path(plan['mechanism_reference_run'])/'models/reliability_detail/initialization.json').read_text())
    assert all(reference_initial[k]==historical[k] for k in reference_initial),'Historical initialization changed'
    reference_state={k:v.detach().clone() for k,v in reference.state_dict().items()}
    apply_runtime(reference,runtime)
    with torch.no_grad():reference_logits=reference(images)
    del reference;gc.collect()
    records={}
    for variant,flags in plan['variants'].items():
        configure(plan['seed'])
        model=build_model(plan,variant).eval()
        initial=initialization_digest(model)
        assert initial['cpu_rng_sha256']==reference_initial['cpu_rng_sha256']
        static=flags.get('reliability_mode')=='static'
        actual=model.state_dict()
        expected_keys={k for k in reference_state if not(static and k.startswith('reliability.'))}
        actual_common={k for k in actual if not(static and k.startswith('reliability.'))}
        assert expected_keys==actual_common
        for key in expected_keys:torch.testing.assert_close(actual[key],reference_state[key],rtol=0,atol=0)
        if not static:assert initial==reference_initial
        apply_runtime(model,runtime)
        with torch.no_grad():output=model(images)
        torch.testing.assert_close(output,reference_logits,rtol=0,atol=0)
        if static:
            captured={}
            token_method=model.tokenize
            def tokenize(x,w):
                captured['token']=w.detach().clone()
                return token_method(x,w)
            def detail_hook(module,args):captured['detail']=args[1].detach().clone()
            model.tokenize=tokenize
            handle=model.spectral_detail.register_forward_pre_hook(detail_hook)
            with torch.no_grad():
                model.reliability.score['logits'].copy_(torch.linspace(-.5,.5,224))
                model(images)
            torch.testing.assert_close(captured['token'],captured['detail'],rtol=0,atol=0)
            torch.testing.assert_close(captured['token'][0],captured['token'][1],rtol=0,atol=0)
            assert (captured['token']-1).abs().max()>.01
            handle.remove();model.tokenize=token_method
            del token_method
        gate=flags.get('detail_use_gate',True);hf=flags.get('detail_use_high_frequency',True)
        assert all(p.requires_grad==gate for p in model.spectral_detail.gate.parameters())
        assert all(p.requires_grad==hf for p in model.spectral_detail.filter.parameters())
        assert all(not p.requires_grad for p in model.network.encoder.parameters())
        records[variant]=dict(initialization=initial,common_tensors_match_reference=True,
            initial_logits_match=True,cpu_rng_matches=True,parameters=model.parameter_report())
        print('PASS seed',plan['seed'],variant,flush=True)
        del model,output,actual;gc.collect()
    result=dict(status='passed',checked_at=now(),seed=plan['seed'],device='cpu',
        reference_initialization=reference_initial,reference_run=plan['mechanism_reference_run'],
        variants=records,unit_checks=checks,
        note='All controls start from the same shared tensors, CPU RNG and initial logits as the historical A+B reference. A static vector is a lower-parameter comparator, not a capacity-matched MLP.')
    shared.save_json(run_dir/'mechanism_preflight.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path);parser.add_argument('--unit-only',action='store_true')
    args=parser.parse_args()
    if args.unit_only:unit_checks()
    else:
        assert args.run_dir is not None
        main(args.run_dir.resolve())
