#!/usr/bin/env python3
"""Prepare immutable sources and run SRPA plus matched ablations sequentially."""
import argparse
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.data import shared,subset_metadata
from src.model import read_pretrained,checkpoint_config
from train import fingerprint,now
from src.experiment_controls import deadline_reached


def resolve_path(value):
    path=Path(value)
    return path if path.is_absolute() else (ROOT/path).resolve()


def prepare(run_dir,config_path):
    run_dir.mkdir(parents=True,exist_ok=True)
    if (run_dir/'protocol.json').exists(): return
    plan=json.loads(config_path.read_text())
    for key in ('pretrained_checkpoint','metadata','text_embeddings','cache_root','manifest_path','scenes_root'):
        if plan.get(key):
            plan[key]=str(resolve_path(plan[key]))
    source=run_dir/'source_snapshot'
    source.mkdir()
    for name in ['src','vendor','configs','scripts']:
        if (ROOT/name).exists():
            shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.log','*.partial'))
    for name in ['train.py','run_experiments.py','infer_scene.py','README.md']:
        if (ROOT/name).exists():
            shutil.copy2(ROOT/name,source/name)
    if (ROOT/'checkpoints').exists():
        (source/'checkpoints').mkdir()
        for name in ['provenance.json','encoder_loading.json']:
            if (ROOT/'checkpoints'/name).exists():
                shutil.copy2(ROOT/'checkpoints'/name,source/'checkpoints'/name)
    cache_root=Path(plan['cache_root'])
    manifest_path=Path(plan['manifest_path']) if plan.get('manifest_path') else None
    if manifest_path and manifest_path.exists():
        info=shared.prepare_cache(run_dir,cache_root)
        rows=list(csv.DictReader(manifest_path.open()))
        valid_pixels={s:sum(int(r['class_0_pixels'])+int(r['class_1_pixels']) for r in rows if r['split']==s) for s in ['tr','val','ts']}
    else:
        import numpy as np
        counts,valid_pixels={},{}
        for s in ('tr','val','ts'):
            labels=np.load(cache_root/f'{s}_labels.npy',mmap_mode='r')
            counts[s]=int(len(labels))
            valid_pixels[s]=int((labels!=255).sum())
        info=dict(manifest_sha256=None,counts=counts,scenes={'tr':[],'val':[],'ts':[]})
    metadata=json.loads(Path(plan['metadata']).read_text())
    indices=list(range(224)) if plan['band_indices'] is None else plan['band_indices']
    metadata=subset_metadata(metadata,indices)
    plan.update(created_at=now(),bands=len(indices),band_indices=indices,sensor_metadata=metadata,
        text_embeddings=str(source/'configs/text_embeddings.pt'),
        pretrained_sha256=fingerprint(plan['pretrained_checkpoint']),
        pretrained_config=checkpoint_config(read_pretrained(plan['pretrained_checkpoint'])),
        source_manifest_sha256=info['manifest_sha256'],counts=info['counts'],scenes=info['scenes'],
        valid_pixels=valid_pixels,
        loss={'name':'CrossEntropyLoss','reduction':'valid_pixel_mean','ignore_index':255,'class_weights':None,'auxiliary_loss':None},
        formulas={'reliability':'w_c = C * softmax(s)_c','routing':'pi_l = softmax(router_l(condition))','adaptation':'F_prime = F + sum_k pi_lk * A_lk(F)'},
        frozen_modules=['SpecAware dynamic spectral embedding','SpecAware ViT blocks, class token and final norm'],
        trainable_modules=['reliability estimator','condition network','routers','adapters','neck','UPerNet decoder','segmentation head'],
        selection='maximum validation mIoU; no test-based selection',test_only_after_training=True,
        normalization='per-tile per-band minmax',
    )
    if plan.get('model_family')=='srpa_innovations_v2':
        plan['formulas'].update(
            local_reliability='w_nc = C * softmax_c(s_c + 0.5*tanh(delta_nc)); zero-initialized local scores',
            spatial_adapter='F_prime = F + up([h_cls, h_grid + GELU(pi_1*DWConv3(h_grid) + pi_2*DWConv3_dilation2(h_grid))])',
            spectral_detail='logits = UPerNet_logits + zero_init_conv(detail(X*w_detail) * sigmoid(gate(decoded)))')
        plan['trainable_modules']=['enabled global reliability','enabled full-resolution spectral detail skip',
                                   'neck','UPerNet decoder','segmentation head']
    runtime=plan.get('runtime_defaults')
    if plan.get('training_class_weights') is not None:
        plan['loss']['class_weights']=plan['training_class_weights']
        plan['loss']['reduction']='weighted_sum_divided_by_valid_pixel_count'
    if plan.get('oil_dice_weight',0):
        plan['loss']['auxiliary_loss']=dict(name='oil_soft_dice',weight=plan['oil_dice_weight'],
            formula='1-(2*sum(p*y)+1)/(sum(p)+sum(y)+1)',ignore_index=255,
            reduction='mean over oil-positive images in full effective batch; zero if none',
            background_images='trained by cross entropy only',microbatch_invariant=True)
        beta=plan.get('oil_tversky_fn_weight',0.5)
        assert 0 < beta < 1
        if beta != 0.5 or plan.get('oil_tversky_schedule'):
            plan['loss']['auxiliary_loss'].update(name='oil_soft_tversky',fn_weight=beta,fp_weight=1-beta,
                formula='1-(TP+0.5)/(TP+(1-beta)*FP+beta*FN+0.5)')
        if plan.get('oil_tversky_schedule'):
            from src.experiment_controls import training_tversky_fn_weight
            schedule=plan['oil_tversky_schedule']
            assert schedule['end_update'] <= plan['max_steps']
            training_tversky_fn_weight(1,beta,schedule)
            plan['loss']['auxiliary_loss'].update(fn_weight_schedule=schedule,
                fn_weight_note='fn_weight is the default; schedule overrides it at each one-based optimizer update')
    if not plan.get('evaluate_test',True):
        plan['test_only_after_training']=False
        plan['test_evaluation']='deferred; training and selection read training/validation splits only'
    if runtime:
        runtime=dict(runtime)
        assert runtime.get('effective_batch',plan['batch_size'])==plan['batch_size']
        assert plan['batch_size'] % runtime.get('micro_batch',plan['micro_batch'])==0
        if runtime.get('micro_batch',plan['micro_batch'])!=plan['micro_batch']:
            assert runtime.get('ghost_batch_size')==plan['micro_batch']
        runtime.update(source_directory=str(source.resolve()))
        shared.save_json(run_dir/'runtime.json',runtime)
    shared.save_json(run_dir/'protocol.json',plan)
    hashes={str(p.relative_to(source)):fingerprint(p) for p in source.rglob('*') if p.is_file()}
    shared.save_json(source/'source_sha256.json',hashes)
    print('PREPARED '+str(run_dir),flush=True)


def comparison(run_dir,variants):
    rows=[]
    for variant in variants:
        p=run_dir/'models'/variant/'PR_mIoU_IoU_percent.csv'
        if p.exists(): rows.extend(csv.DictReader(p.open()))
    if rows:
        with (run_dir/'comparison_percent.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shared.save_json(run_dir/'comparison.json',rows)


def refresh_reports(run_dir,plan):
    script=plan.get('report_script')
    if script:
        with (run_dir/'report_update.log').open('a') as log:
            result=subprocess.run([sys.executable,script],stdout=log,stderr=subprocess.STDOUT,timeout=120)
        shared.save_json(run_dir/'report_update.json',dict(updated_at=now(),returncode=result.returncode,script=script))


def manager(args):
    run_dir=args.run_dir.resolve()
    prepare(run_dir,args.config)
    plan=json.loads((run_dir/'protocol.json').read_text())
    variants=args.variants or list(plan['variants'])
    assert all(v in plan['variants'] for v in variants)
    runtime_path=args.runtime_config
    if runtime_path is None and (run_dir/'runtime.json').exists(): runtime_path=run_dir/'runtime.json'
    runtime=json.loads(runtime_path.read_text()) if runtime_path else {}
    source=args.source_dir or Path(runtime.get('source_directory',run_dir/'source_snapshot'))
    runner=source/'train.py'
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    state=dict(status='starting',manager_pid=os.getpid(),run_directory=str(run_dir),variants=variants,completed=[],updated_at=now())
    state.update(source_directory=str(source),runtime_config=None if runtime_path is None else str(runtime_path))
    shared.save_json(ROOT/'current_run.json',dict(manager_pid=os.getpid(),run_directory=str(run_dir),pipeline=str(run_dir/'pipeline.json'),started_at=now(),variants=variants))
    refresh_reports(run_dir,plan)

    def scene_predictions(variant):
        if args.skip_scene_predictions or not plan.get('scenes',{}).get('ts'): return
        scenes_root=Path(plan['scenes_root']) if plan.get('scenes_root') else None
        if scenes_root is None or not scenes_root.exists(): return
        folder=run_dir/'predictions'/variant
        folder.mkdir(parents=True,exist_ok=True)
        for scene in plan['scenes']['ts']:
            scene_file=scenes_root/(scene+'.mat')
            if not scene_file.exists(): continue
            output=folder/scene
            if Path(str(output)+'.metadata.json').exists(): continue
            command=[sys.executable,'-u',str(source/'infer_scene.py'),
                '--run-dir',str(run_dir),'--variant',variant,
                '--scene',str(scene_file),
                '--output',str(output)]
            with (folder/(scene+'.console.log')).open('a') as log:
                child=subprocess.Popen(command,cwd=source,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='scene_prediction',active_variant=variant,active_scene=scene,worker_pid=child.pid,updated_at=now())
            shared.save_json(run_dir/'pipeline.json',state)
            if child.wait():
                state.update(status='failed',console_log=str(folder/(scene+'.console.log')),updated_at=now())
                shared.save_json(run_dir/'pipeline.json',state)
                raise RuntimeError(f'Scene prediction failed: {variant} {scene}')
        state.pop('active_scene',None)

    for variant in variants:
        if deadline_reached(getattr(args,'stop_at',None) or plan.get('stop_at')):
            state.update(status='paused',reason='deadline',active_variant=None,worker_pid=None,updated_at=now())
            shared.save_json(run_dir/'pipeline.json',state)
            return 75
        if (run_dir/'models'/variant/'result.json').exists():
            result=json.loads((run_dir/'models'/variant/'result.json').read_text())
            if result.get('status')=='complete':
                scene_predictions(variant)
                state['completed'].append(variant);continue
        for smoke in [True,False]:
            if deadline_reached(getattr(args,'stop_at',None) or plan.get('stop_at')):
                state.update(status='paused',reason='deadline',active_variant=variant,worker_pid=None,updated_at=now())
                shared.save_json(run_dir/'pipeline.json',state)
                return 75
            folder=run_dir/(('smoke_performance' if runtime else 'smoke') if smoke else 'models')/variant
            if smoke and (folder/'result.json').exists() and json.loads((folder/'result.json').read_text()).get('status')=='passed': continue
            folder.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-u',str(runner),'--run-dir',str(run_dir),'--variant',variant]
            if getattr(args,'stop_at',None): command.extend(['--stop-at',args.stop_at])
            if runtime_path: command.extend(['--runtime-config',str(runtime_path)])
            if smoke: command.append('--smoke')
            with (folder/'console.log').open('a') as log:
                child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='preflight' if smoke else 'training',active_variant=variant,worker_pid=child.pid,progress_file=str(folder/'progress.json'),updated_at=now())
            shared.save_json(run_dir/'pipeline.json',state)
            print(json.dumps(dict(event='launch',variant=variant,smoke=smoke,pid=child.pid)),flush=True)
            while child.poll() is None:
                state['updated_at']=now();shared.save_json(run_dir/'pipeline.json',state)
                time.sleep(5)
            if child.returncode:
                if child.returncode==75:
                    state.update(status='paused',returncode=75,reason='checkpoint_saved_at_deadline',worker_pid=None,updated_at=now())
                    shared.save_json(run_dir/'pipeline.json',state)
                    return 75
                state.update(status='failed',returncode=child.returncode,console_log=str(folder/'console.log'),updated_at=now())
                shared.save_json(run_dir/'pipeline.json',state)
                raise RuntimeError(f'{variant} failed during {"smoke" if smoke else "training"}: {folder / "console.log"}')
        comparison(run_dir,variants)
        refresh_reports(run_dir,plan)
        scene_predictions(variant)
        state['completed'].append(variant)
    state.update(status='complete',active_variant=None,worker_pid=None,updated_at=now())
    shared.save_json(run_dir/'pipeline.json',state)
    comparison(run_dir,variants)
    refresh_reports(run_dir,plan)
    print('ALL EXPERIMENTS COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/selected_spectral_detail.json')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--variants',nargs='+')
    parser.add_argument('--skip-scene-predictions',action='store_true')
    parser.add_argument('--runtime-config',type=Path)
    parser.add_argument('--source-dir',type=Path)
    parser.add_argument('--stop-at')
    args=parser.parse_args()
    if args.prepare_only:
        prepare(args.run_dir.resolve(),args.config)
    else:
        sys.exit(manager(args))
