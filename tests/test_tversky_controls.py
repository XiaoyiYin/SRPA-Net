"""Regression, formula, ignored pixels and accumulation checks for recall loss."""
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiment_controls import oil_dice_loss_sum, oil_tversky_loss_sum


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    labels=torch.tensor([[[1,0,255],[0,1,0]],[[0,0,255],[0,0,0]],
                         [[1,1,0],[255,0,0]],[[255,255,255],[255,255,255]]])
    logits=torch.randn(4,2,2,3,requires_grad=True)
    assert torch.equal(oil_dice_loss_sum(logits,labels),oil_tversky_loss_sum(logits,labels))
    for beta,coefficient in [(.5,.25),(.6,.25),(.7,.25),(.8,.25),(.7,.5)]:
        valid=labels!=255; positive=(labels==1).flatten(1).any(1).sum()
        def loss(x,y):
            return (torch.nn.functional.cross_entropy(x,y,ignore_index=255,reduction='sum')/valid.sum()
                    +coefficient*oil_tversky_loss_sum(x,y,beta)/positive)
        z=logits.detach().clone().requires_grad_(); full=loss(z,labels); full.backward()
        sliced=logits.detach().clone().requires_grad_()
        parts=sum(loss(x,y) for x,y in zip(sliced.split(1),labels.split(1))); parts.backward()
        torch.testing.assert_close(full,parts); torch.testing.assert_close(z.grad,sliced.grad)
        assert (z.grad.permute(0,2,3,1)[~valid]==0).all()
        changed=z.detach().clone(); changed.permute(0,2,3,1)[~valid]=1000
        torch.testing.assert_close(oil_tversky_loss_sum(changed,labels,beta),oil_tversky_loss_sum(z,labels,beta))
        manual=0
        for probabilities,y in zip(z.softmax(1)[:,1],labels):
            if not (y==1).any():continue
            tp=probabilities[y==1].sum(); fp=probabilities[y==0].sum(); fn=(1-probabilities[y==1]).sum()
            manual+=1-(tp+.5)/(tp+(1-beta)*fp+beta*fn+.5)
        torch.testing.assert_close(oil_tversky_loss_sum(z,labels,beta),manual)
        perfect=torch.stack(((labels!=1).float()*40,(labels==1).float()*40),1)
        assert oil_tversky_loss_sum(perfect,labels,beta)<1e-6
        empty=z.detach().clone().requires_grad_()
        value=oil_tversky_loss_sum(empty,torch.zeros_like(labels),beta); value.backward()
        assert value==0 and (empty.grad==0).all()
    # Greater beta increases the relative gradient emphasis on missed oil.
    ratios=[]
    for beta in [.5,.6,.7,.8]:
        z=torch.zeros(1,2,1,2,requires_grad=True)
        oil_tversky_loss_sum(z,torch.tensor([[[1,0]]]),beta).backward()
        ratios.append(float(-z.grad[0,1,0,0]/z.grad[0,1,0,1]))
    assert ratios==sorted(ratios) and len(set(ratios))==4
    for beta in [0,1,float('nan')]:
        try:oil_tversky_loss_sum(logits,labels,beta)
        except ValueError:pass
        else:raise AssertionError('Invalid beta accepted')
    print('PASS: exact default Dice, manual formula, ignored/empty/perfect labels, accumulated loss/gradients, relative FN gradient, invalid beta')


if __name__=='__main__':main()
