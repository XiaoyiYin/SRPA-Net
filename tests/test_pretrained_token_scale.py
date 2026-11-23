"""Compare the actual pretrained tokenizer with sum-C weighted aggregation."""
from pathlib import Path
import sys,json
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.model import SRPANet
from src.data import OilSpillTiles


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    plan=json.loads((ROOT/'configs/experiment.json').read_text())
    model=SRPANet(plan['pretrained_checkpoint'],json.loads(Path(plan['metadata']).read_text()),plan['text_embeddings']).cuda().eval()
    image=OilSpillTiles(plan['cache_root'],'tr')[0][0].unsqueeze(0).cuda()
    with torch.no_grad(),torch.autocast('cuda',enabled=False):
        reference,_=model.network.encoder.patch_embed(image,model.wavelength_nm[None],model.fwhm_nm[None],model.sensor_name,model.data_level)
        weighted=model.tokenize(image,torch.ones(1,224,device='cuda'))
    torch.testing.assert_close(reference,weighted,atol=1e-5,rtol=1e-4)
    print('PASS: actual frozen pretrained tokenizer scale restored; max error',float((reference-weighted).abs().max()))


if __name__=='__main__': main()
