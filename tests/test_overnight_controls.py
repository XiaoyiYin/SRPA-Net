"""Small CPU checks for branch gradients, schedules, loss and deadline semantics."""
from datetime import datetime
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.innovations import SRPAInnovations, SpectralDetail
from src.experiment_controls import deadline_reached, reliability_schedule_scale, optimizer_parameters


class Reliability(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Parameter(torch.linspace(-1, 1, 7))

    def forward(self, x, wavelengths, fwhm):
        scores = self.score.expand(x.shape[0], -1)
        return 7*scores.softmax(-1), None, scores


class Neck(nn.Module):
    def forward(self, features):
        return features


class Decoder(nn.Module):
    def forward(self, *features):
        return features[-1]


def miniature():
    model = SRPAInnovations.__new__(SRPAInnovations)
    nn.Module.__init__(model)
    model.config = dict(img_size=8, patch_size=4, embed_dim=8, depth=12)
    model.wavelength_nm = torch.arange(7)
    model.fwhm_nm = torch.ones(7)
    model.gsd = torch.tensor([10.])
    model.use_reliability = True
    model.use_local_reliability = model.use_spatial_adapter = False
    model.use_spectral_detail = True
    model.detail_use_reliability = True
    model.detail_detach_weights = model.detail_detach_context = False
    model.reliability_strength = model.reliability_schedule_scale = 1.
    model.activation_checkpointing = False
    model.reliability = Reliability()
    model.network = nn.Module()
    model.network.encoder = nn.Module()
    model.network.encoder.cls_token = nn.Parameter(torch.zeros(1, 1, 8), requires_grad=False)
    model.network.encoder.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
    model.network.encoder.norm = nn.Identity()
    model.network.neck = Neck()
    model.network.decoder = Decoder()
    model.network.segmentation_head = nn.Conv2d(8, 2, 1)
    # Produce full-resolution logits as in the real model.
    model.network.segmentation_head = nn.Sequential(model.network.segmentation_head, nn.Upsample(size=(8,8)))
    model.spectral_detail = SpectralDetail(7, decoder_dim=8, width=4)
    nn.init.normal_(model.spectral_detail.output.weight, std=.1)
    model.position_encoding = lambda dim, grid, gsd, cls_token: torch.zeros(len(gsd), 5, 8)
    def tokenize(x, weights):
        pooled = F.avg_pool2d((x*weights[:,:,None,None]).sum(1,keepdim=True),4)
        return pooled.flatten(2).transpose(1,2).expand(-1,-1,8)
    model.tokenize = tokenize
    return model


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    x = torch.rand(2,7,8,8)
    model = miniature()
    captured = {}
    def capture(module, inputs):
        captured.update(weights=inputs[1], context=inputs[2])
    model.spectral_detail.register_forward_pre_hook(capture)
    reference = 7*model.reliability.score.softmax(-1)
    for alpha in [0.,.25,.5,1.]:
        model.reliability_strength = alpha
        output = model(x)
        torch.testing.assert_close(captured['weights'], (1+alpha*(reference-1)).expand(2,-1))
        torch.testing.assert_close(captured['weights'].sum(-1), torch.full((2,),7.))
        assert torch.isfinite(output).all()
    model.reliability_strength = 1.
    for detach_weights,detach_context in [(False,False),(True,False),(False,True),(True,True)]:
        model.detail_detach_weights = detach_weights
        model.detail_detach_context = detach_context
        model.zero_grad(set_to_none=True)
        output = model(x)
        assert captured['weights'].requires_grad != detach_weights
        assert captured['context'].requires_grad != detach_context
        # The main branch still trains reliability even with BOTH skip paths detached.
        output.square().mean().backward()
        grad = model.reliability.score.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
        assert model.spectral_detail.gate.weight.grad.abs().sum()>0
        assert model.spectral_detail.projection.weight.grad.abs().sum()>0
    schedule = dict(delay_steps=2000,ramp_steps=4000)
    assert [reliability_schedule_scale(s,schedule) for s in [0,2000,4000,6000,20000]] == [0,0,.5,1,1]
    model.reliability_schedule_scale = 0
    model.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    assert model.reliability.score.grad.abs().sum()==0
    groups = optimizer_parameters(model,1e-4,.1)
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids))==len([p for p in model.parameters() if p.requires_grad])
    assert groups[0]['lr']==1e-4 and abs(groups[1]['lr']-1e-5)<1e-12
    # Ignore pixels contribute zero; weighted loss keeps the valid-pixel denominator.
    logits = torch.tensor([[[[2.,1.,0.]],[[0.,2.,3.]]]],requires_grad=True)
    labels = torch.tensor([[[0,1,255]]])
    plain = F.cross_entropy(logits,labels,ignore_index=255,reduction='none')
    weighted = F.cross_entropy(logits,labels,weight=torch.tensor([1.,1.5]),ignore_index=255,reduction='sum')/2
    torch.testing.assert_close(weighted,(plain[0,0,0]+1.5*plain[0,0,1])/2)
    deadline = '2026-10-21T08:00:00+08:00'
    assert not deadline_reached(deadline,datetime.fromisoformat('2026-10-20T23:59:59+00:00'))
    assert deadline_reached(deadline,datetime.fromisoformat('2026-10-21T00:00:00+00:00'))
    print('PASS: strength/sum-C, actual branch detach routing, retained main/gate/projection gradients, warmup, LR groups, weighted CE, Shanghai deadline')


if __name__ == '__main__':
    main()
