#!/usr/bin/env python3
"""Tile and stitch an HSI scene, preserving original channel/pixel coordinates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from train import build_model,configure,fingerprint
from src.runtime import apply_runtime


def tile_positions(height,width,size=224):
    for top in range(0,height,size):
        for left in range(0,width,size):
            yield top,left,min(size,height-top),min(size,width-left)


def normalize_tile(tile):
    # Exact same per-band float64 min-max rule as the baseline dataset reader.
    tile=tile.astype(np.float64)
    for band in range(tile.shape[0]):
        minimum=tile[band].min();span=tile[band].max()-minimum
        if span!=0: tile[band]=(tile[band]-minimum)/(span+1e-8)
    return tile.astype(np.float32)


def read_cube(path,key='img'):
    if path.suffix.lower()=='.mat':
        cube=loadmat(path,variable_names=[key])[key]
        assert cube.ndim==3
        return cube.transpose(2,0,1),None
    if path.suffix.lower()=='.npy':
        cube=np.load(path,mmap_mode='r')
        assert cube.ndim==3,'NPY input must be C,H,W'
        return cube,None
    from osgeo import gdal
    ds=gdal.OpenEx(str(path),gdal.OF_RASTER|gdal.OF_READONLY,open_options=['NUM_THREADS=1'])
    if ds is None: raise OSError(path)
    cube=ds.ReadAsArray()
    geo=dict(projection=ds.GetProjection(),transform=ds.GetGeoTransform(can_return_null=True))
    return cube,geo


@torch.no_grad()
def predict_scene(model,cube,band_indices,output,threshold=.5):
    _,height,width=cube.shape
    output.parent.mkdir(parents=True,exist_ok=True)
    probability=np.lib.format.open_memmap(str(output)+'.probability.npy',mode='w+',dtype='float32',shape=(height,width))
    mask=np.lib.format.open_memmap(str(output)+'.mask.npy',mode='w+',dtype='uint8',shape=(height,width))
    model.eval()
    with Path(str(output)+'.tiles.jsonl').open('w') as log:
        for top,left,h,w in tile_positions(height,width):
            tile=cube[band_indices,top:top+h,left:left+w]
            tile=np.pad(tile,((0,0),(0,224-h),(0,224-w)),mode='edge')
            x=torch.from_numpy(normalize_tile(tile)).unsqueeze(0).cuda()
            with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
            if not torch.isfinite(logits).all():
                with torch.autocast('cuda',enabled=False): logits=model(x.float())
            assert torch.isfinite(logits).all()
            p=logits.float().softmax(1)[0,1,:h,:w].cpu().numpy()
            probability[top:top+h,left:left+w]=p
            mask[top:top+h,left:left+w]=(p>threshold).astype(np.uint8)
            diagnostics=model.last_diagnostics
            record=dict(top=top,left=left,height=h,width=w,
                band_weights=diagnostics['weights'][0].cpu().tolist(),
                routes=None if diagnostics['routing'] is None else diagnostics['routing'][0].cpu().tolist())
            local=diagnostics.get('local_weights')
            if local is not None:
                local_path=Path(str(output)+f'.tile_{top}_{left}.local_weights.npy')
                np.save(local_path,local[0].float().cpu().numpy())
                record.update(local_band_weights=str(local_path),local_grid_shape=[28,28],local_order='row-major token, original band')
            log.write(json.dumps(record)+'\n')
    probability.flush();mask.flush()
    return dict(height=height,width=width,threshold=threshold,mask_rule='oil_probability > threshold',
        outputs=[str(output)+'.probability.npy',str(output)+'.mask.npy'],
        interpretation='Segmentation probabilities; not calibrated uncertainty or oil thickness.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--variant',default='srpa_full')
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--mat-key',default='img')
    args=parser.parse_args()
    assert 0<=args.threshold<=1
    plan=json.loads((args.run_dir/'protocol.json').read_text())
    configure(plan['seed'])
    assert fingerprint(plan['pretrained_checkpoint'])==plan['pretrained_sha256']
    model=build_model(plan,args.variant).cuda()
    if (args.run_dir/'runtime.json').exists():
        apply_runtime(model,json.loads((args.run_dir/'runtime.json').read_text()))
    checkpoint=args.checkpoint or args.run_dir/'models'/args.variant/'best.ckpt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['protocol_sha256']==fingerprint(args.run_dir/'protocol.json')
    model.load_state_dict(saved['model'],strict=True)
    if hasattr(model,'reliability_schedule_scale'):
        model.reliability_schedule_scale=saved.get('reliability_schedule_scale',1.0)
    cube,geo=read_cube(args.scene,args.mat_key)
    assert cube.shape[0]>max(plan['band_indices'])
    result=predict_scene(model,cube,plan['band_indices'],args.output,args.threshold)
    result.update(scene=str(args.scene),checkpoint=str(checkpoint),metadata=plan['sensor_metadata'],
                  georeferencing=geo,band_indices=plan['band_indices'])
    Path(str(args.output)+'.metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
