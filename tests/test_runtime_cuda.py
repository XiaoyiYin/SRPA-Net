"""Check real-checkpoint gradient agreement and deferred overflow recovery."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train import build_model,configure
from src.model import RangeSafeConv2d
from src.runtime import apply_runtime
from src.data import OilSpillTiles,shared


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads((args.run_dir/'protocol.json').read_text())
    runtime=json.loads((args.run_dir/'runtime.json').read_text())
    state=torch.load(args.run_dir/'pre_optimization_step1000.ckpt',map_location='cpu',weights_only=False)
    data=OilSpillTiles(plan['cache_root'],'tr',plan['band_indices'])
    indices=next(iter(shared.StepBatches(len(data),8,1000,1001,42)))
    images=torch.stack([data[index][0] for index in indices])
    labels=torch.stack([data[index][1] for index in indices])
    images,labels=shared.gpu_augmentation(images,labels,1000,42,8)
    count=int((labels!=255).sum())
    outputs=[];gradients=[];losses=[]
    for fast in [False,True]:
        configure(42)
        model=build_model(plan,'combined_v2').cuda()
        model.load_state_dict(state['model'])
        if fast: apply_runtime(model,runtime)
        model.train()
        logits=[];loss_value=0.
        micro=8 if fast else 4
        for x,y in zip(images.split(micro),labels.split(micro)):
            with torch.autocast('cuda',dtype=torch.float16):
                output=model(x)
                loss=torch.nn.functional.cross_entropy(output.float(),y,ignore_index=255,reduction='sum')/count
            assert torch.isfinite(loss)
            (loss*1024).backward()
            logits.append(output.detach().float().cpu());loss_value+=float(loss)
        assert all(p.grad is None for p in model.network.encoder.parameters())
        grad=torch.cat([p.grad.detach().float().cpu().flatten()/1024 for p in model.parameters() if p.grad is not None])
        assert torch.isfinite(grad).all()
        gradients.append(grad);outputs.append(torch.cat(logits));losses.append(loss_value)
        del model,grad
        gc.collect();torch.cuda.empty_cache()
    a,b=gradients[0].double(),gradients[1].double()
    relative=float((a-b).norm()/a.norm())
    cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0))
    difference=float((outputs[0]-outputs[1]).abs().max())
    assert abs(losses[0]-losses[1])<1e-4,(losses,relative)
    assert relative<.02 and cosine>.999,relative
    assert difference<.05,difference
    # Force real FP16 convolution overflow, then restore BN state and repeat
    # in FP32 as the training loop does. Compare to a clean FP32 reference.
    conv=RangeSafeConv2d(2,2,1,bias=False).cuda()
    torch.nn.init.constant_(conv.weight,4.)
    network=torch.nn.Sequential(conv,torch.nn.BatchNorm2d(2).cuda()).train()
    clean=copy.deepcopy(network)
    conv.defer_range_check=True
    x=torch.full((4,2,8,8),20000.,device='cuda')
    saved={k:v.clone() for k,v in network.named_buffers()}
    with torch.autocast('cuda',dtype=torch.float16): bad=network(x)
    assert not torch.isfinite(bad).all()
    with torch.no_grad():
        for k,v in network.named_buffers(): v.copy_(saved[k])
    with torch.autocast('cuda',enabled=False):
        recovered=network(x);expected=clean(x)
    assert torch.isfinite(recovered).all()
    torch.testing.assert_close(recovered,expected,rtol=0,atol=0)
    for k,v in network.named_buffers():
        torch.testing.assert_close(v,dict(clean.named_buffers())[k],rtol=0,atol=0)
    result=dict(status='passed',checkpoint_step=1000,reference_loss=losses[0],accelerated_loss=losses[1],
        gradient_relative_l2=relative,gradient_cosine=cosine,max_logit_difference=difference,
        frozen_encoder_gradients_none=True,forced_fp16_overflow_fp32_recovery='passed')
    (args.run_dir/'runtime_equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
