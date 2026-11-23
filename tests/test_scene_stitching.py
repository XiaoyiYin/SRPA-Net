"""Check full-scene coverage, edge cropping, band ordering and pixel locations."""
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infer_scene import predict_scene,tile_positions


class CoordinateOracle(torch.nn.Module):
    def forward(self,x):
        # A deliberately reordered constant-band cube must arrive as [33,11].
        assert x.shape==(1,2,224,224)
        assert torch.all(x[:,0]==33) and torch.all(x[:,1]==11)
        rows=torch.arange(224,device=x.device)[:,None]
        cols=torch.arange(224,device=x.device)[None,:]
        logit=rows/100-cols/200
        self.last_diagnostics={'weights':torch.ones(1,2,device=x.device),'routing':None}
        return torch.stack((torch.zeros_like(logit),logit),0).unsqueeze(0)


def main():
    height,width=225,449
    coverage=np.zeros((height,width),dtype=np.uint8)
    for top,left,h,w in tile_positions(height,width): coverage[top:top+h,left:left+w]+=1
    assert np.all(coverage==1)
    cube=np.empty((3,height,width),dtype=np.int16)
    for band,value in enumerate([11,22,33]): cube[band]=value
    with tempfile.TemporaryDirectory(prefix='srpa-stitch-') as tmp:
        output=Path(tmp)/'oracle'
        predict_scene(CoordinateOracle().cuda(),cube,[2,0],output,.5)
        probability=np.load(str(output)+'.probability.npy')
        mask=np.load(str(output)+'.mask.npy')
        rows=np.arange(height)[:,None]%224
        cols=np.arange(width)[None,:]%224
        expected=1/(1+np.exp(-(rows/100-cols/200)))
        np.testing.assert_allclose(probability,expected,atol=1e-6)
        np.testing.assert_array_equal(mask,(expected>.5).astype(np.uint8))
    print('PASS: scene pixel coordinates, full coverage, non-square edge tiles, band order and threshold masks')


if __name__=='__main__': main()
