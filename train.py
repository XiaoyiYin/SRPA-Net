#!/usr/bin/env python3
"""Controlled SRPA-Net oil-spill experiments with resumable optimizer updates."""
import argparse
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.model import SRPANet
from src.data import OilSpillTiles,shared,subset_metadata
from src.runtime import apply_runtime,set_optimizer_runtime
from src.experiment_controls import (deadline_reached,reliability_schedule_scale,optimizer_parameters,
                                     set_training_lr,oil_tversky_loss_sum,training_tversky_fn_weight)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def fingerprint(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def configure(seed):
    torch.set_num_threads(2)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True


def build_model(plan,variant):
    model_class=SRPANet
    if plan.get('model_family')=='srpa_innovations_v2':
        from src.innovations import SRPAInnovations
        model_class=SRPAInnovations
    elif plan.get('model_family')=='srpa_mechanism_controls':
        from src.mechanism_controls import SRPAMechanismControls
        model_class=SRPAMechanismControls
    elif plan.get('model_family')=='srpa_weighting_location':
        from src.weighting_location import SRPAWeightingLocation
        model_class=SRPAWeightingLocation
    elif plan.get('model_family')=='srpa_shared_weighting_location':
        from src.shared_weighting_location import SRPASharedWeightingLocation
        model_class=SRPASharedWeightingLocation
    return model_class(plan['pretrained_checkpoint'],plan['sensor_metadata'],plan['text_embeddings'],
        **plan['variants'][variant],adapter_rank=plan['adapter_rank'],experts=plan['experts'],
        activation_checkpointing=plan['activation_checkpointing'])


def diagnostic(model):
    weights=model.last_diagnostics['weights'].float()
    routes=model.last_diagnostics['routing']
    result=dict(weight_min=float(weights.min()),weight_max=float(weights.max()),
        weight_sum_max_error=float((weights.sum(-1)-weights.shape[-1]).abs().max()),
        route_sum_max_error=None if routes is None else float((routes.sum(-1)-1).abs().max()))
    local=model.last_diagnostics.get('local_weights')
    if local is not None:
        result.update(local_weight_min=float(local.min()),local_weight_max=float(local.max()),
            local_weight_sum_max_error=float((local.sum(-1)-local.shape[-1]).abs().max()),
            local_delta_abs_max=float(model.last_diagnostics['local_delta'].abs().max()),
            spatial_weight_std_mean=float(local.std(1,unbiased=False).mean()))
    return result


def frozen_digest(model):
    digest=hashlib.sha256()
    for name,p in model.network.encoder.named_parameters():
        digest.update(name.encode());digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def initialization_digest(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return dict(state_sha256=digest.hexdigest(),
                cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())


@torch.no_grad()
def evaluate(model,batches,micro,limit=None):
    model.eval()
    matrix=torch.zeros(2,2,dtype=torch.int64,device='cuda')
    loss_sum=torch.zeros((),device='cuda');count=0
    for batch_index,(images,labels) in enumerate(batches):
        if limit is not None and batch_index>=limit: break
        for x,y in zip(images.split(micro),labels.split(micro)):
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16): output=model(x)
            if not torch.isfinite(output).all():
                with torch.autocast('cuda',enabled=False):
                    output=torch.cat([model(part.float()) for part in x.split(getattr(model,'fp32_retry_micro_batch',micro))])
            if not torch.isfinite(output).all(): raise FloatingPointError('Nonfinite evaluation logits')
            loss_sum+=nn.functional.cross_entropy(output.float(),y,ignore_index=255,reduction='sum')
            valid=y!=255;count+=int(valid.sum())
            prediction=output.argmax(1)
            matrix+=torch.bincount(2*y[valid]+prediction[valid],minlength=4).reshape(2,2)
    result=shared.scores(matrix)
    result['loss']=float(loss_sum/max(1,count))
    return result


def run(args):
    run_dir=args.run_dir.resolve()
    plan=json.loads((run_dir/'protocol.json').read_text())
    stop_at=getattr(args,'stop_at',None) or plan.get('stop_at')
    stop_requested=False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGUSR1,request_stop)
    runtime_path=args.runtime_config
    if runtime_path is None and (run_dir/'runtime.json').exists(): runtime_path=run_dir/'runtime.json'
    runtime=json.loads(runtime_path.read_text()) if runtime_path else {}
    runtime_hash=fingerprint(runtime_path) if runtime_path else None
    variant=args.variant
    smoke=args.smoke
    smoke_folder='smoke_performance' if runtime else 'smoke'
    out=run_dir/(smoke_folder if smoke else 'models')/variant
    out.mkdir(parents=True,exist_ok=True)
    for rel,expected in json.loads((ROOT/'source_sha256.json').read_text()).items():
        assert fingerprint(ROOT/rel)==expected,rel
    assert fingerprint(plan['pretrained_checkpoint'])==plan['pretrained_sha256']
    assert fingerprint('/data/SpectralX-main/oilspill_prepared/manifest.csv')==plan['source_manifest_sha256']
    configure(plan['seed'])
    model=build_model(plan,variant)
    if plan.get('verify_factorial_initialization'):
        initial=initialization_digest(model)
        expected=json.loads((run_dir/'factorial_preflight.json').read_text())
        assert expected['status']=='passed'
        assert initial==expected['initialization'], 'Factorial initialization/RNG differs from preflight'
        shared.save_json(out/'initialization.json',dict(variant=variant,**initial))
    if plan.get('verify_mechanism_initialization'):
        initial=initialization_digest(model)
        expected=json.loads((run_dir/'mechanism_preflight.json').read_text())
        assert expected['status']=='passed'
        assert initial==expected['variants'][variant]['initialization'], 'Mechanism control initialization/RNG differs from preflight'
        shared.save_json(out/'initialization.json',dict(variant=variant,**initial))
    if plan.get('verify_weighting_initialization'):
        initial=initialization_digest(model)
        expected=json.loads((run_dir/'weighting_preflight.json').read_text())
        assert expected['status']=='passed'
        assert initial==expected['initialization'], 'Weighting control initialization/RNG differs from preflight'
        shared.save_json(out/'initialization.json',dict(variant=variant,**initial))
    model=model.cuda()
    apply_runtime(model,runtime)
    optimizer=torch.optim.AdamW(optimizer_parameters(model,plan['lr'],plan.get('reliability_lr_multiplier',1.0)),lr=plan['lr'],betas=tuple(plan['betas']),weight_decay=plan['weight_decay'],fused=runtime.get('fused_adamw',False))
    class_weights=plan.get('training_class_weights')
    dice_weight=plan.get('oil_dice_weight',0.0)
    assert 0 <= dice_weight <= 1
    tversky_fn_weight=plan.get('oil_tversky_fn_weight',0.5)
    assert 0 < tversky_fn_weight < 1
    if class_weights is not None:
        assert len(class_weights)==2 and all(w>0 for w in class_weights)
        class_weights=torch.tensor(class_weights,dtype=torch.float32,device='cuda')
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    step,best,best_step=0,-1.,0
    protocol_hash=fingerprint(run_dir/'protocol.json')
    if not smoke and (out/'last.ckpt').exists():
        state=torch.load(out/'last.ckpt',map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==protocol_hash and state['variant']==variant
        model.load_state_dict(state['model'],strict=True)
        if hasattr(model,'reliability_schedule_scale'):
            model.reliability_schedule_scale=state.get('reliability_schedule_scale',1.0)
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        step,best,best_step=state['step'],state['best_mIoU'],state['best_step']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        del state;gc.collect()
    set_optimizer_runtime(optimizer,runtime)
    start_step=step
    target=3 if smoke else plan['max_steps']
    micro=args.micro_batch if args.micro_batch else runtime.get('micro_batch',plan['micro_batch'])
    assert plan['batch_size']%micro==0 and micro>=2
    if micro!=plan['micro_batch']:
        assert runtime.get('ghost_batch_size')==plan['micro_batch'],'Preserve the original BatchNorm group size'
    train_data=OilSpillTiles(plan['cache_root'],'tr',plan['band_indices'])
    sampler=shared.StepBatches(len(train_data),plan['batch_size'],step,target,plan['seed'])
    train_loader=shared.loader(train_data,plan['batch_size'],plan['workers'],sampler)
    val_loader=shared.loader(OilSpillTiles(plan['cache_root'],'val',plan['band_indices']),plan['batch_size'],plan['workers'])
    report=model.parameter_report()
    shared.save_json(out/'model.json',dict(variant=variant,checkpoint_config=model.config,pretrained_loading=model.pretrained_loading,
        parameters=report,micro_batch=micro,batchnorm_micro_batch=runtime.get('ghost_batch_size',micro),
        effective_batch=plan['batch_size'],runtime=runtime,runtime_sha256=runtime_hash,
        GPU=torch.cuda.get_device_name(0),torch=torch.__version__))
    if runtime:
        shared.save_json(out/f'runtime_start_step{step}.json',dict(step=step,runtime=runtime,runtime_sha256=runtime_hash,at=now()))
    frozen_before=frozen_digest(model)
    print(json.dumps(dict(event='started',variant=variant,step=step,target=target,parameters=report)),flush=True)
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    gradient_checks={}

    def persist():
        shared.checkpoint(out/'last.ckpt',dict(step=step,best_mIoU=best,best_step=best_step,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),protocol_sha256=protocol_hash,variant=variant,
            frozen_encoder_sha256=frozen_before,runtime_sha256=runtime_hash,runtime=runtime,
            reliability_schedule_scale=getattr(model,'reliability_schedule_scale',1.0)))

    def pause_if_requested():
        if stop_requested or deadline_reached(stop_at):
            persist()
            shared.save_json(out/'progress.json',dict(status='paused',variant=variant,
                step=step,target=target,best_mIoU=best,best_step=best_step,
                reason='deadline_or_stop_signal',checkpoint=str(out/'last.ckpt'),updated_at=now()))
            print('PAUSED_WITH_CHECKPOINT',flush=True)
            raise SystemExit(75)

    for images,labels in train_loader:
        pause_if_requested()
        set_training_lr(optimizer,step+1,plan['max_steps'],plan.get('training_lr_schedule'))
        current_fn_weight=training_tversky_fn_weight(step+1,tversky_fn_weight,plan.get('oil_tversky_schedule'))
        if hasattr(model,'reliability_schedule_scale'):
            model.reliability_schedule_scale=reliability_schedule_scale(step+1,plan.get('reliability_schedule'))
        model.train()
        images,labels=shared.gpu_augmentation(images,labels,step,plan['seed'],plan['batch_size'])
        valid_total=int((labels!=255).sum());assert valid_total>0
        oil_positive_total=int((labels==1).flatten(1).any(1).sum()) if dice_weight else 0
        rng=torch.cuda.get_rng_state()
        buffers={n:b.detach().clone() for n,b in model.named_buffers() if any(k in n for k in ('running_mean','running_var','num_batches_tracked'))}
        fp32_retry=False
        for attempt in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss_value=0.;bad_forward=False
            attempt_micro=min(micro,getattr(model,'fp32_retry_micro_batch',micro)) if fp32_retry else micro
            for x,y in zip(images.split(attempt_micro),labels.split(attempt_micro)):
                with torch.autocast('cuda',dtype=torch.float16,enabled=not fp32_retry):
                    output=model(x)
                    assert output.shape==(x.shape[0],2,224,224),output.shape
                    loss=nn.functional.cross_entropy(output.float(),y,weight=class_weights,ignore_index=255,reduction='sum')/valid_total
                    if dice_weight and oil_positive_total:
                        loss=loss+dice_weight*oil_tversky_loss_sum(output,y,current_fn_weight)/oil_positive_total
                if not torch.isfinite(loss):
                    if fp32_retry: raise FloatingPointError(f'FP32 loss nonfinite at {step}')
                    bad_forward=True;fp32_retry=True;break
                scaler.scale(loss).backward();loss_value+=float(loss.detach())
            if not bad_forward:
                previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()>=previous: break
            print(json.dumps(dict(event='numeric_retry',step=step,attempt=attempt+1,fp32=fp32_retry)),flush=True)
            torch.cuda.set_rng_state(rng)
            with torch.no_grad():
                for name,buf in model.named_buffers():
                    if name in buffers: buf.copy_(buffers[name])
        else:
            raise FloatingPointError('Persistent nonfinite gradients')
        step+=1
        if smoke:
            assert all(p.grad is None for p in model.network.encoder.parameters())
            groups=[('reliability_score',model.reliability.score),('quality_encoder',model.reliability.quality),
                    ('routers',model.routers),('adapters',model.adapters)]
            if hasattr(model,'innovation_groups'): groups.extend(model.innovation_groups().items())
            for name,module in groups:
                gradients=[p.grad.detach().float() for p in module.parameters() if p.grad is not None]
                assert all(torch.isfinite(g).all() for g in gradients)
                gradient_checks[name]=sum(float(g.abs().sum()) for g in gradients)
        if smoke or step<=3 or step%10==0:
            record=dict(status='training',variant=variant,step=step,target=target,train_loss=loss_value,
                best_mIoU=best,best_step=best_step,elapsed_seconds=time.monotonic()-started,
                seconds_per_update=(time.monotonic()-started)/max(1,step-start_step),peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
                updated_at=now(),reliability=diagnostic(model))
            record['learning_rates']=[group['lr'] for group in optimizer.param_groups]
            if dice_weight:
                record['overlap_fn_weight']=current_fn_weight
            shared.save_json(out/'progress.json',record)
            print(json.dumps(record),flush=True)
        if not smoke and (step%plan['val_interval']==0 or step==target):
            validation=evaluate(model,val_loader,micro)
            assert validation['valid_pixels']==plan['valid_pixels']['val']
            record=dict(step=step,validation=validation,updated_at=now())
            with (out/'validation_history.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
            if validation['mIoU']>best:
                best,best_step=validation['mIoU'],step
                shared.checkpoint(out/'best.ckpt',dict(step=step,model=model.state_dict(),validation=validation,
                    variant=variant,protocol_sha256=protocol_hash,
                    reliability_schedule_scale=getattr(model,'reliability_schedule_scale',1.0)))
            persist()
            shared.save_json(out/'progress.json',dict(status='training',variant=variant,step=step,target=target,best_mIoU=best,best_step=best_step,validation=validation,updated_at=now()))
            print(json.dumps(record),flush=True)
        elif not smoke and (step==10 or step==start_step+10 or step%plan['checkpoint_interval']==0): persist()
        pause_if_requested()
    assert frozen_digest(model)==frozen_before,'Frozen encoder weights changed'
    if smoke:
        if plan['variants'][variant]['use_reliability']:
            if getattr(model,'reliability_schedule_scale',1.0)>0:
                assert gradient_checks['reliability_score']>0,gradient_checks
            else:
                assert gradient_checks['reliability_score']==0,gradient_checks
        else:
            assert gradient_checks['reliability_score']==0,gradient_checks
            if not plan['variants'][variant]['use_adapters']:
                assert gradient_checks['quality_encoder']==0,gradient_checks
            assert torch.equal(model.last_diagnostics['weights'],torch.ones_like(model.last_diagnostics['weights']))
        if plan['variants'][variant]['use_adapters']:
            assert gradient_checks['routers']>0 and gradient_checks['adapters']>0,gradient_checks
        if hasattr(model,'innovation_groups'):
            for name,module in model.innovation_groups().items():
                if any(p.requires_grad for p in module.parameters()):
                    assert gradient_checks[name]>0,(name,gradient_checks)
        validation=evaluate(model,val_loader,micro)
        assert validation['valid_pixels']==plan['valid_pixels']['val']
        persist()
        restored=torch.load(out/'last.ckpt',map_location='cpu',weights_only=False)
        model.load_state_dict(restored['model'],strict=True)
        optimizer.load_state_dict(restored['optimizer']);scaler.load_state_dict(restored['scaler'])
        result=dict(status='passed',variant=variant,updates=step,micro_batch=micro,effective_batch=plan['batch_size'],
            full_validation=True,test_evaluated=False,validation=validation,gradient_checks=gradient_checks,
            frozen_encoder_unchanged=True,checkpoint_roundtrip=True,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
            elapsed_seconds=time.monotonic()-started,parameters=report)
        result.update(runtime=runtime,runtime_sha256=runtime_hash,batchnorm_micro_batch=runtime.get('ghost_batch_size',micro))
        shared.save_json(out/'result.json',result)
        print('SMOKE PASSED '+json.dumps(result),flush=True)
        return
    pause_if_requested()
    if not plan.get('evaluate_test',True):
        selected=torch.load(out/'best.ckpt',map_location='cpu',weights_only=False)
        result=dict(status='complete',variant=variant,step=step,best_step=best_step,
            best_val_mIoU=best,validation_best=selected['validation'],
            test_evaluated=False,parameters=report,frozen_encoder_unchanged=True,
            elapsed_seconds=time.monotonic()-started,completed_at=now(),
            runtime=runtime,runtime_sha256=runtime_hash,
            batchnorm_micro_batch=runtime.get('ghost_batch_size',micro))
        shared.save_json(out/'result.json',result)
        shared.save_json(out/'progress.json',dict(status='complete',variant=variant,
            step=step,target=target,test_evaluated=False,updated_at=now()))
        print('COMPLETE_VALIDATION_ONLY '+json.dumps(result),flush=True)
        return
    test_loader=shared.loader(OilSpillTiles(plan['cache_root'],'ts',plan['band_indices']),plan['batch_size'],plan['workers'])
    final=evaluate(model,test_loader,micro)
    selected=torch.load(out/'best.ckpt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'],strict=True)
    if hasattr(model,'reliability_schedule_scale'):
        model.reliability_schedule_scale=selected.get('reliability_schedule_scale',1.0)
    del selected;gc.collect()
    best_test=evaluate(model,test_loader,micro)
    assert final['valid_pixels']==best_test['valid_pixels']==plan['valid_pixels']['ts']
    result=dict(status='complete',variant=variant,step=step,best_step=best_step,best_val_mIoU=best,
        test_final=final,test_best_validation=best_test,parameters=report,frozen_encoder_unchanged=True,
        elapsed_seconds=time.monotonic()-started,completed_at=now())
    result.update(runtime=runtime,runtime_sha256=runtime_hash,batchnorm_micro_batch=runtime.get('ghost_batch_size',micro))
    shared.save_json(out/'result.json',result)
    with (out/'PR_mIoU_IoU_percent.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['model','selection','step','P_percent','R_percent','F1_percent','mIoU_percent','oil_IoU_percent','TN','FP','FN','TP'])
        writer.writeheader()
        for selection,metrics,s in [('best_validation',best_test,best_step),('final',final,step)]:
            tn,fp=metrics['confusion_matrix'][0];fn,tp=metrics['confusion_matrix'][1]
            writer.writerow(dict(model=variant,selection=selection,step=s,P_percent=metrics['oil_precision']*100,
                R_percent=metrics['oil_recall']*100,F1_percent=metrics['oil_F1']*100,mIoU_percent=metrics['mIoU']*100,oil_IoU_percent=metrics['oil_IoU']*100,TN=tn,FP=fp,FN=fn,TP=tp))
    shared.save_json(out/'progress.json',dict(status='complete',variant=variant,step=step,target=target,updated_at=now()))
    print('COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--variant',default='srpa_full')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--micro-batch',type=int)
    parser.add_argument('--runtime-config',type=Path)
    parser.add_argument('--stop-at',help='ISO timestamp with timezone; overrides protocol deadline')
    args=parser.parse_args()
    try: run(args)
    except Exception as error:
        runtime_active=args.runtime_config is not None or (args.run_dir/'runtime.json').exists()
        out=args.run_dir/(('smoke_performance' if runtime_active else 'smoke') if args.smoke else 'models')/args.variant
        out.mkdir(parents=True,exist_ok=True)
        shared.save_json(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),at=now()))
        progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
        progress.update(status='failed',error=repr(error),updated_at=now())
        shared.save_json(out/'progress.json',progress)
        raise
