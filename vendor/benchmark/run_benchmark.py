#!/usr/bin/env python3
"""Run seven full-band oil-spill baselines in the isolated SpectralX environment."""
import argparse
import csv
from datetime import datetime
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

HERE = Path(__file__).resolve().parent
PROJECT = Path('/data/SpectralX-main')
ENV = Path('/home/chic/miniconda3/envs/SpectralX')
sys.path.insert(0, str(HERE))
from models import NAMES, build_model, model_description


def now():
    return datetime.now().isoformat()


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_cache(run, cache):
    """Read/verify every original image and label, retaining exact WHUOHS normalization."""
    root = PROJECT / 'oilspill_prepared'
    manifest = root / 'manifest.csv'
    rows = list(csv.DictReader(manifest.open(newline='')))
    counts = {s: sum(r['split'] == s for r in rows) for s in ('tr', 'val', 'ts')}
    assert counts == {'tr': 262, 'val': 64, 'ts': 76}, counts
    scenes = {s: sorted({r['scene'] for r in rows if r['split'] == s}) for s in counts}
    assert all(set(scenes[a]).isdisjoint(scenes[b]) for a, b in [('tr','val'),('tr','ts'),('val','ts')])
    backup = json.loads((PROJECT / 'backups/20261012_203bands_before_retraining/backup_manifest.json').read_text())
    assert backup['status'] == 'verified'
    references = {v['path']: v for v in backup['files']}
    fingerprint = sha256(manifest)
    cache.mkdir(parents=True, exist_ok=True)
    completion = cache / 'complete.json'
    if completion.exists():
        info = json.loads(completion.read_text())
        assert info['manifest_sha256'] == fingerprint and info['bands'] == 224
        # A prior complete cache is reusable only if its source files are unchanged.
        unchanged = all((root / r['image']).stat().st_mtime_ns == info['source_stats'][r['image']]
                        and (root / r['label']).stat().st_mtime_ns == info['source_stats'][r['label']]
                        for r in rows)
        if unchanged:
            save_json(run / 'data_provenance.json', info)
            return info
    whu = load_source(PROJECT / 'src/datasets/WHU_OHS.py', 'original_whu_reader')
    total, source_stats, checked = len(rows), {}, 0
    for split in counts:
        selected = [r for r in rows if r['split'] == split]
        images = np.lib.format.open_memmap(cache / f'{split}_images.npy', mode='w+', dtype='float32',
                                          shape=(len(selected), 224, 224, 224))
        labels = np.lib.format.open_memmap(cache / f'{split}_labels.npy', mode='w+', dtype='uint8',
                                          shape=(len(selected), 224, 224))
        dataset = whu.WHUOHS([str(root / r['image']) for r in selected],
                            [str(root / r['label']) for r in selected])
        for i, row in enumerate(selected):
            for key in ('image', 'label'):
                rel = row[key]
                path = root / rel
                reference = references['oilspill_prepared/' + rel]
                assert path.stat().st_size == reference['bytes'] and sha256(path) == reference['sha256'], path
                source_stats[rel] = path.stat().st_mtime_ns
            sample = dataset[i]
            image, mask = sample['image'].numpy(), sample['mask'].squeeze(0).numpy()
            assert image.shape == (224, 224, 224) and np.isfinite(image).all()
            assert set(np.unique(mask)) <= {0, 1, 255}
            assert int((mask == 0).sum()) == int(row['class_0_pixels'])
            assert int((mask == 1).sum()) == int(row['class_1_pixels'])
            assert int((mask == 255).sum()) == int(row['ignore_pixels'])
            images[i], labels[i] = image, mask
            checked += 1
            if checked % 20 == 0 or checked == total:
                save_json(run / 'cache_progress.json', dict(status='preparing', checked=checked, total=total))
                print(f'Cache and SHA256 verification: {checked}/{total}', flush=True)
        images.flush(); labels.flush()
        del images, labels
        save_json(cache / f'{split}_rows.json', selected)
    info = dict(status='verified', bands=224, counts=counts, scenes=scenes, source_root=str(root),
                cache_root=str(cache), manifest_sha256=fingerprint, source_stats=source_stats,
                normalization='original_WHUOHS_per_tile_per_band_minmax_float32',
                original_tiff_sha256_verified=True, completed_at=now(), original_data_modified=False)
    save_json(completion, info)
    save_json(run / 'data_provenance.json', info)
    return info


class CachedDataset(Dataset):
    def __init__(self, cache, split, augment=False):
        self.images = np.load(Path(cache) / f'{split}_images.npy', mmap_mode='r')
        self.labels = np.load(Path(cache) / f'{split}_labels.npy', mmap_mode='r')
        # Frozen copy of precisely the same spatial augmentation used by SpectralX.
        self.transform = load_source(HERE / 'original_augmentations.py', 'original_augmentations').get_train_augmentation(224, seg_fill=255) if augment else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            index, seed = key
            random.seed(seed); torch.manual_seed(seed)
        else:
            index = key
        image = torch.from_numpy(self.images[index].copy())
        mask = torch.from_numpy(self.labels[index].copy()).unsqueeze(0)
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask.squeeze(0).long()


class StepBatches(Sampler):
    """Identical scene order and augmentation seeds across architectures and resumes."""
    def __init__(self, count, batch_size, start, stop, seed):
        self.count, self.batch_size, self.start, self.stop, self.seed = count, batch_size, start, stop, seed

    def __iter__(self):
        per_epoch = self.count // self.batch_size
        last_epoch, order = -1, None
        for step in range(self.start, self.stop):
            epoch, batch = divmod(step, per_epoch)
            if epoch != last_epoch:
                order = torch.randperm(self.count, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
                last_epoch = epoch
            yield [(idx, self.seed + step * self.batch_size + j)
                   for j, idx in enumerate(order[batch * self.batch_size:(batch + 1) * self.batch_size])]

    def __len__(self):
        return self.stop - self.start


def loader(dataset, batch_size, workers, batch_sampler=None):
    kwargs = dict(dataset=dataset, num_workers=workers, pin_memory=True)
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    if batch_sampler is not None:
        kwargs['batch_sampler'] = batch_sampler
    else:
        kwargs.update(batch_size=batch_size, shuffle=False, drop_last=False)
    return DataLoader(**kwargs)


def gpu_augmentation(images, labels, step, seed, batch_size):
    """Same per-sample random draws, interpolation and fill as the original transform.

    GPU placement is the only change; all 224 bands share one spatial transform.
    """
    from torchvision.transforms import functional as TF
    images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
    transformed_images, transformed_masks = [], []
    for j, (image, mask) in enumerate(zip(images, labels)):
        rng = random.Random(seed + step * batch_size + j)
        if rng.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if rng.random() < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        angle = rng.random() * 180 - 90
        if rng.random() < 0.5:
            image = TF.rotate(image, angle, TF.InterpolationMode.BILINEAR, expand=False, fill=0)
            mask = TF.rotate(mask.unsqueeze(0), angle, TF.InterpolationMode.NEAREST,
                             expand=False, fill=255).squeeze(0)
        transformed_images.append(image)
        transformed_masks.append(mask)
    return torch.stack(transformed_images), torch.stack(transformed_masks)


def scores(matrix):
    m = matrix.double().cpu()
    tp = m.diag(); actual = m.sum(1); predicted = m.sum(0)
    iou = tp / (actual + predicted - tp).clamp_min(1)
    precision = tp / predicted.clamp_min(1); recall = tp / actual.clamp_min(1)
    f1 = 2 * tp / (actual + predicted).clamp_min(1)
    return dict(OA=float(tp.sum() / m.sum().clamp_min(1)), mIoU=float(iou.mean()),
                mF1=float(f1.mean()), oil_IoU=float(iou[1]), oil_precision=float(precision[1]),
                oil_recall=float(recall[1]), oil_miss_rate=float(1-recall[1]), oil_F1=float(f1[1]),
                per_class_IoU=iou.tolist(), confusion_matrix=m.long().tolist(), valid_pixels=int(m.sum()))


@torch.no_grad()
def evaluate(model, batches, micro_batch, limit=None):
    model.eval()
    matrix = torch.zeros((2, 2), dtype=torch.int64, device='cuda')
    loss_sum = torch.zeros((), device='cuda'); pixels = torch.zeros((), device='cuda')
    for batch_index, (images, labels) in enumerate(batches):
        if limit is not None and batch_index >= limit:
            break
        for x, y in zip(images.split(micro_batch), labels.split(micro_batch)):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                output = model(x)
            logits = output.float()
            loss_sum += nn.functional.cross_entropy(logits, y, ignore_index=255, reduction='sum')
            valid = y != 255
            pixels += valid.sum()
            prediction = logits.argmax(1)
            matrix += torch.bincount(2 * y[valid] + prediction[valid], minlength=4).reshape(2, 2)
    result = scores(matrix)
    result['loss'] = float(loss_sum / pixels.clamp_min(1))
    return result


def checkpoint(path, state):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(state, handle)
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def worker(args, smoke=False):
    plan = json.loads((args.run_dir / 'protocol.json').read_text())
    sys.path.insert(0, plan['smp_source'])
    import segmentation_models_pytorch as smp
    assert str(Path(smp.__file__).resolve()).startswith(plan['smp_source'])
    torch.set_num_threads(2)
    torch.manual_seed(plan['seed']); np.random.seed(plan['seed']); random.seed(plan['seed'])
    torch.cuda.manual_seed_all(plan['seed'])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_model(args.model).cuda().to(memory_format=torch.channels_last)
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    assert first_conv.in_channels == 224
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['lr'], betas=(0.9, 0.95), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    output = args.run_dir / ('smoke' if smoke else 'models') / args.model
    output.mkdir(parents=True, exist_ok=True)
    started, step, best, best_step = time.monotonic(), 0, -1., 0
    if not smoke and (output / 'last.ckpt').exists():
        saved = torch.load(output / 'last.ckpt', map_location='cpu', weights_only=False)
        assert saved['protocol_sha256'] == sha256(args.run_dir / 'protocol.json')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        step, best, best_step = saved['step'], saved['best_mIoU'], saved['best_step']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved; gc.collect()
    max_steps = 2 if smoke else plan['max_steps']
    data = CachedDataset(plan['cache_root'], 'tr', augment=False)
    batch_sampler = StepBatches(len(data), plan['batch_size'], step, max_steps, plan['seed'])
    train = loader(data, plan['batch_size'], plan['workers'], batch_sampler)
    val = loader(CachedDataset(plan['cache_root'], 'val'), plan['batch_size'], plan['workers'])
    test = loader(CachedDataset(plan['cache_root'], 'ts'), plan['batch_size'], plan['workers'])
    micro = plan['micro_batch']
    description = model_description(args.model)
    description.update(parameters=sum(p.numel() for p in model.parameters()), smp_version=smp.__version__)
    save_json(output / 'model.json', description)
    print(json.dumps(dict(event='started', model=args.model, step=step, target=max_steps, **description)), flush=True)

    def persist_last():
        checkpoint(output / 'last.ckpt', dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), best_mIoU=best, best_step=best_step,
                    protocol_sha256=sha256(args.run_dir / 'protocol.json'), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), architecture=args.model))

    torch.cuda.reset_peak_memory_stats()
    for images, labels in train:
        model.train()
        images, labels = gpu_augmentation(images, labels, step, plan['seed'], plan['batch_size'])
        total_valid = (labels != 255).sum().item()
        if not total_valid:
            raise ValueError('All-ignore training batch')
        # A skipped AMP update does not consume a data batch or optimizer step.
        rng = torch.cuda.get_rng_state()
        buffers = {n:b.detach().clone() for n,b in model.named_buffers()
                   if any(t in n for t in ('running_mean','running_var','num_batches_tracked'))}
        for attempt in range(16):
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.
            for x, y in zip(images.split(micro), labels.split(micro)):
                x, y = x.cuda(non_blocking=True).to(memory_format=torch.channels_last), y.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(x)
                    assert logits.shape == (x.shape[0], 2, 224, 224), logits.shape
                    # Weight micro-batch means by VALID PIXELS, reproducing a single
                    # CrossEntropyLoss(ignore_index=255, reduction='mean') over the full batch.
                    fraction = (y != 255).sum() / total_valid
                    loss = criterion(logits.float(), y) * fraction
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at step {step}')
                scaler.scale(loss).backward(); train_loss += float(loss.detach())
            scale_before = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= scale_before:
                break
            print(f'AMP skipped non-finite gradients at step {step}; rerun same batch', flush=True)
            torch.cuda.set_rng_state(rng)
            with torch.no_grad():
                for name, buffer in model.named_buffers():
                    if name in buffers: buffer.copy_(buffers[name])
        else:
            raise FloatingPointError('AMP gradients remain non-finite after 16 attempts')
        step += 1
        if smoke or step % 50 == 0:
            record = dict(status='training', model=args.model, step=step, target=max_steps, train_loss=train_loss,
                          elapsed_seconds=time.monotonic()-started, updated_at=now(),
                          peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
            save_json(output / 'progress.json', record)
            print(json.dumps(record), flush=True)
        if not smoke and (step % plan['val_interval'] == 0 or step == max_steps):
            metrics = evaluate(model, val, micro)
            assert metrics['valid_pixels'] == plan['valid_pixels']['val']
            record = dict(step=step, validation=metrics, updated_at=now())
            with (output / 'validation_history.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            if metrics['mIoU'] > best:
                best, best_step = metrics['mIoU'], step
                checkpoint(output / 'best.ckpt', dict(step=step, model=model.state_dict(),
                           validation=metrics, architecture=args.model, protocol_sha256=sha256(args.run_dir / 'protocol.json')))
            persist_last()
            save_json(output / 'progress.json', dict(status='training', model=args.model, step=step,
                      target=max_steps, best_step=best_step, best_mIoU=best, validation=metrics, updated_at=now()))
            print(json.dumps(record), flush=True)
        elif not smoke and step % plan['checkpoint_interval'] == 0:
            persist_last()
    if smoke:
        assert step == max_steps, f'Only {step} of {max_steps} smoke updates applied'
        result = dict(status='passed', optimizer_updates=step, full_data_evaluation=False,
                      validation=evaluate(model, val, micro, limit=1),
                      test=evaluate(model, test, micro, limit=1),
                      peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30, model=description)
        assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
        save_json(output / 'result.json', result)
        return
    final_test = evaluate(model, test, micro)
    assert final_test['valid_pixels'] == plan['valid_pixels']['ts']
    selected = torch.load(output / 'best.ckpt', map_location='cpu', weights_only=False)
    model.load_state_dict(selected['model']); del selected; gc.collect()
    best_test = evaluate(model, test, micro)
    assert best_test['valid_pixels'] == plan['valid_pixels']['ts']
    result = dict(status='complete', model=args.model, step=step, best_step=best_step, best_val_mIoU=best,
                  test_final=final_test, test_best_validation=best_test, description=description,
                  completed_at=now(), elapsed_seconds=time.monotonic()-started)
    save_json(output / 'result.json', result)
    save_json(output / 'progress.json', dict(status='complete', model=args.model, step=step, target=max_steps, updated_at=now()))
    print('COMPLETE '+json.dumps(result), flush=True)


def comparison(run, names):
    rows = []
    for name in names:
        p = run / 'models' / name / 'result.json'
        if not p.exists():
            continue
        result = json.loads(p.read_text())
        for selection, key in [('best_validation', 'test_best_validation'), ('final', 'test_final')]:
            metric = result[key]
            rows.append(dict(model=name, selection=selection, training_step=result['best_step'] if selection=='best_validation' else result['step'],
                             **{k:metric[k] for k in ('OA','mIoU','oil_IoU','oil_precision','oil_recall','oil_miss_rate','oil_F1')}))
    if rows:
        with (run / 'comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save_json(run / 'comparison.json', rows)


def manager(args):
    run = args.run_dir.resolve(); run.mkdir(parents=True, exist_ok=True)
    if not (run / 'protocol.json').exists():
        snapshot = run / 'code_snapshot'
        snapshot.mkdir()
        for name in ('run_benchmark.py','models.py','README.md','check_protocol.py'):
            shutil.copy2(HERE / name, snapshot / name)
        shutil.copy2(PROJECT / 'src/datamodules/augmentations.py', snapshot / 'original_augmentations.py')
        shutil.copytree(args.smp_source / 'segmentation_models_pytorch', snapshot / 'smp/segmentation_models_pytorch',
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        cache = args.cache_root.resolve()
        info = prepare_cache(run, cache)
        save_json(run/'cache_progress.json',dict(status='verified',checked=402,total=402,bands=224))
        manifest_rows = list(csv.DictReader((PROJECT / 'oilspill_prepared/manifest.csv').open(newline='')))
        plan = dict(created_at=now(), models=NAMES, bands=224, classes=[0,1], ignore_index=255,
                    loss=dict(name='CrossEntropyLoss', ignore_index=255, reduction='mean', class_weights=None,
                              label_smoothing=0., auxiliary_loss=None),
                    seed=42, pretrained=False, lr=1e-4, optimizer='AdamW', betas=[0.9,0.95], weight_decay=1e-4,
                    lr_scheduler=None, batch_size=args.batch_size, micro_batch=args.micro_batch, max_steps=args.max_steps,
                    precision='fp16_mixed', spatial_augmentation='original_transform_equivalent_on_GPU',
                    val_interval=500, checkpoint_interval=500, workers=args.workers,
                    max_parallel=args.parallel, cache_root=str(cache),
                    smp_source=str(snapshot/'smp'), description={n:model_description(n) for n in NAMES},
                    source_manifest_sha256=info['manifest_sha256'],
                    valid_pixels={s:sum(int(r['class_0_pixels'])+int(r['class_1_pixels']) for r in manifest_rows if r['split']==s) for s in ('tr','val','ts')},
                    original_data_modified=False, selection='max_validation_mIoU; test only after all training updates',
                    comparison_limitations='One seed; different backbones; effective batch 8 differs from historical SpectralX batch 2; early DeepLab variants are CNN-only adaptations.')
        save_json(run / 'protocol.json', plan)
        save_json(snapshot / 'sha256.json', {str(p.relative_to(snapshot)):sha256(p) for p in snapshot.rglob('*.py')})
    plan = json.loads((run / 'protocol.json').read_text())
    runner = run/'code_snapshot/run_benchmark.py'
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    def launch(name, mode):
        folder = run / ('smoke' if mode == 'smoke' else 'models') / name
        folder.mkdir(parents=True, exist_ok=True)
        log = (folder / 'console.log').open('a')
        child = subprocess.Popen([sys.executable, '-u', str(runner), '--run-dir', str(run),
                                  '--mode', mode, '--model', name], cwd=PROJECT, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        return child
    state = dict(status='smoke_testing', run_directory=str(run), manager_pid=os.getpid(),
                 models={n:dict(status='pending') for n in NAMES}, updated_at=now())
    for name in NAMES:
        result_path = run/'smoke'/name/'result.json'
        if not result_path.exists():
            print(f'Smoke testing {name}: train/backward/checkpoint-compatible metrics/val/test, all 224 channels', flush=True)
            state.update(smoke_model=name, updated_at=now()); save_json(run/'pipeline.json',state)
            child = launch(name, 'smoke')
            code = child.wait()
            if code:
                raise RuntimeError(f'Smoke test failed: {name}; exit {code}; see {run / "smoke" / name / "console.log"}')
        smoke = json.loads(result_path.read_text())
        assert smoke['status'] == 'passed'
        # Reserve CUDA context, optimizer/evaluation workspaces and concurrent-process headroom.
        state['models'][name].update(smoke='passed', estimated_gpu_gib=max(2.,smoke['peak_cuda_gib']*1.3+0.8))
    state.update(status='training', updated_at=now()); save_json(run/'pipeline.json',state)
    pending, active, failed = list(NAMES), {}, []
    capacity = torch.cuda.get_device_properties(0).total_memory/2**30 - 2.5
    while pending or active:
        for name, child in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            p = run/'models'/name/'result.json'
            state['models'][name].update(status='complete' if code==0 and p.exists() else 'failed', returncode=code, ended_at=now())
            if state['models'][name]['status']=='failed': failed.append(name)
            del active[name]
            comparison(run,NAMES)
        for name in list(pending):
            if len(active) >= plan['max_parallel']:
                break
            used = sum(state['models'][n]['estimated_gpu_gib'] for n in active)
            needed = state['models'][name]['estimated_gpu_gib']
            if active and used+needed > capacity:
                continue
            if (run/'models'/name/'result.json').exists():
                state['models'][name]['status']='complete'; pending.remove(name); continue
            child = launch(name, 'worker'); active[name]=child; pending.remove(name)
            state['models'][name].update(status='running', pid=child.pid, started_at=now())
            print(f'Started {name}, PID {child.pid}; active={list(active)}',flush=True)
        state.update(active_models=list(active), pending_models=pending.copy(), updated_at=now())
        save_json(run/'pipeline.json',state)
        if pending or active: time.sleep(5)
    comparison(run,NAMES)
    state.update(status='complete' if not failed else 'completed_with_failures', failed_models=failed, completed_at=now())
    save_json(run/'pipeline.json',state)
    print(json.dumps(state),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=['manager','worker','smoke'],default='manager')
    parser.add_argument('--model',choices=NAMES)
    parser.add_argument('--smp-source',type=Path,default=Path('/data/segmentation_models.pytorch-main'))
    parser.add_argument('--cache-root',type=Path,default=Path('/home/chic/oilspill_benchmark_cache/224_fullbands'))
    parser.add_argument('--max-steps',type=int,default=20000)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--micro-batch',type=int,default=8)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--parallel',type=int,default=2)
    args=parser.parse_args()
    if Path(sys.prefix).resolve()!=ENV: parser.error('Use the isolated SpectralX conda environment')
    if not torch.cuda.is_available(): parser.error('CUDA GPU required')
    if args.micro_batch<2 or args.batch_size % args.micro_batch or min(args.max_steps,args.parallel)<1:
        parser.error('Need micro_batch>=2, batch_size divisible by micro_batch, positive steps/parallel')
    try:
        if args.mode=='manager': manager(args)
        else: worker(args,smoke=args.mode=='smoke')
    except BaseException as error:
        traceback.print_exc()
        if args.mode=='manager':
            args.run_dir.mkdir(parents=True,exist_ok=True)
            save_json(args.run_dir/'manager_error.json',dict(error=str(error),traceback=traceback.format_exc(),failed_at=now()))
            p=args.run_dir/'pipeline.json'
            state=json.loads(p.read_text()) if p.exists() else {}
            state.update(status='failed',error=str(error),failed_at=now())
            save_json(p,state)
        raise


if __name__=='__main__':
    main()
