#!/usr/bin/env python3
"""Evaluate an already selected checkpoint without altering training artifacts."""
import argparse
import json
from pathlib import Path
import sys
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run=args.run_dir.resolve()
    source=run/'source_snapshot'
    sys.path.insert(0,str(source))
    from train import build_model,configure,evaluate,fingerprint,now
    from src.data import OilSpillTiles,shared
    from src.runtime import apply_runtime
    plan=json.loads((run/'protocol.json').read_text())
    runtime=json.loads((run/'runtime.json').read_text())
    for rel,want in json.loads((source/'source_sha256.json').read_text()).items():
        assert fingerprint(source/rel)==want,rel
    assert fingerprint(plan['pretrained_checkpoint'])==plan['pretrained_sha256']
    assert fingerprint('/data/SpectralX-main/oilspill_prepared/manifest.csv')==plan['source_manifest_sha256']
    checkpoint=run/'models'/args.variant/'best.ckpt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['protocol_sha256']==fingerprint(run/'protocol.json')
    assert saved['variant']==args.variant
    configure(plan['seed'])
    model=build_model(plan,args.variant).cuda()
    apply_runtime(model,runtime)
    model.load_state_dict(saved['model'],strict=True)
    if hasattr(model,'reliability_schedule_scale'):
        model.reliability_schedule_scale=saved.get('reliability_schedule_scale',1.0)
    micro=runtime.get('micro_batch',plan['micro_batch'])
    validation=evaluate(model,shared.loader(OilSpillTiles(plan['cache_root'],'val',plan['band_indices']),plan['batch_size'],plan['workers']),micro)
    assert validation['valid_pixels']==plan['valid_pixels']['val']
    assert abs(validation['mIoU']-saved['validation']['mIoU'])<1e-5,(validation,saved['validation'])
    test=evaluate(model,shared.loader(OilSpillTiles(plan['cache_root'],'ts',plan['band_indices']),plan['batch_size'],plan['workers']),micro)
    assert test['valid_pixels']==plan['valid_pixels']['ts']
    result=dict(status='complete',evaluated_at=now(),run_directory=str(run),variant=args.variant,
                seed=plan['seed'],checkpoint=str(checkpoint),checkpoint_sha256=fingerprint(checkpoint),
                checkpoint_step=saved['step'],protocol_sha256=fingerprint(run/'protocol.json'),
                validation_reproduced=validation,stored_validation=saved['validation'],test=test,
                checkpoint_selection='maximum validation mIoU, fixed before test evaluation',
                evaluated_with_source_snapshot=str(source),test_reuse='Previously viewed exploratory test split')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    shared.save_json(args.output,result)
    print(json.dumps(dict(seed=plan['seed'],step=saved['step'],validation_mIoU=validation['mIoU'],test=test)),flush=True)


if __name__=='__main__':
    main()
