"""Mathematical invariants and gradient paths, independent of training metrics."""
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model import band_weights,SRPANet,ConditionalAdapters


def main():
    torch.manual_seed(9)
    scores=torch.zeros(2,7,requires_grad=True)
    weights=band_weights(scores)
    torch.testing.assert_close(weights,torch.ones_like(weights))
    torch.testing.assert_close(weights.sum(-1),torch.full((2,),7.))
    (weights*torch.arange(7.)).sum().backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum()>0
    patches=torch.randn(2,5,3,4,dtype=torch.double)
    u=torch.randn(2,3,6,2,dtype=torch.double)
    v=torch.randn(2,3,2,4,dtype=torch.double)
    weights=torch.rand(2,3,dtype=torch.double,requires_grad=True)
    channels=torch.einsum('bncp,bcrp,bcer->bnce',patches,v,u)
    explicit=(channels*weights[:,None,:,None]).sum(2)
    efficient=SRPANet.aggregate(patches,u,v,weights)
    torch.testing.assert_close(explicit,efficient,atol=1e-10,rtol=1e-10)
    reference=channels.sum(2)
    torch.testing.assert_close(SRPANet.aggregate(patches,u,v,torch.ones_like(weights)),reference,atol=1e-10,rtol=1e-10)
    a=torch.autograd.grad(explicit.square().sum(),weights,retain_graph=True)[0]
    b=torch.autograd.grad(efficient.square().sum(),weights)[0]
    torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-8)
    adapter=ConditionalAdapters(dim=12,rank=3,experts=4)
    tokens=torch.randn(2,7,12,requires_grad=True)
    route_logits=torch.randn(2,4,requires_grad=True)
    routes=route_logits.softmax(-1)
    torch.testing.assert_close(routes.sum(-1),torch.ones(2))
    torch.testing.assert_close(adapter(tokens,routes),tokens,atol=0,rtol=0)
    for expert in adapter.experts:
        torch.nn.init.normal_(expert[-1].weight,std=.01)
    adapter(tokens,routes).square().mean().backward()
    assert tokens.grad.shape==tokens.shape and torch.isfinite(tokens.grad).all()
    assert route_logits.grad.abs().sum()>0 and torch.isfinite(route_logits.grad).all()
    print('PASS: sum-C reliability, equal-weight scale, weighted contributions, exact weight gradients, full-space identity adapters and differentiable routes')


if __name__=='__main__': main()
