"""Check loss gradients, ignored labels, microbatch equivalence and LR resume."""
import copy
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiment_controls import oil_dice_loss_sum,training_lr_scale,set_training_lr


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    labels=torch.tensor([[[1,0,255],[0,1,0]],[[0,0,255],[0,0,0]],
                         [[1,1,0],[255,0,0]],[[255,255,255],[255,255,255]]])
    logits=torch.randn(4,2,2,3,requires_grad=True)
    positive=int((labels==1).flatten(1).any(1).sum())
    def loss(x,y):
        return torch.nn.functional.cross_entropy(x,y,ignore_index=255,reduction='sum')/(labels!=255).sum()+.25*oil_dice_loss_sum(x,y)/positive
    full=loss(logits,labels);full.backward();gradient=logits.grad.clone()
    sliced=logits.detach().clone().requires_grad_()
    parts=sum(loss(x,y) for x,y in zip(sliced.split(1),labels.split(1)))
    parts.backward()
    torch.testing.assert_close(full,parts)
    torch.testing.assert_close(gradient,sliced.grad)
    assert torch.all(gradient.permute(0,2,3,1)[labels==255]==0)
    changed=logits.detach().clone()
    changed.permute(0,2,3,1)[labels==255]=1000
    torch.testing.assert_close(oil_dice_loss_sum(changed,labels),oil_dice_loss_sum(logits,labels))
    perfect=torch.stack(((labels!=1).float()*40,(labels==1).float()*40),1)
    assert oil_dice_loss_sum(perfect,labels)<1e-6
    empty=torch.randn(2,2,2,3,requires_grad=True)
    empty_loss=oil_dice_loss_sum(empty,torch.zeros(2,2,3,dtype=torch.long))
    empty_loss.backward();assert empty_loss==0 and torch.all(empty.grad==0)
    # Increasing oil logit at a true oil pixel must decrease the overlap loss.
    z=torch.zeros(1,2,2,3,requires_grad=True)
    oil_dice_loss_sum(z,labels[:1]).backward();assert z.grad[0,1,0,0]<0
    schedule=dict(name='warmup_cosine',warmup_steps=500,warmup_start_ratio=.1,min_lr_ratio=.01)
    assert training_lr_scale(1,20000,schedule)==.1
    assert training_lr_scale(500,20000,schedule)==1
    assert training_lr_scale(20000,20000,schedule)==.01
    assert training_lr_scale(1000,20000,None)==1
    p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([p],lr=1e-4)
    set_training_lr(opt,10000,20000,schedule)
    state=copy.deepcopy(opt.state_dict())
    restored=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=1e-4)
    restored.load_state_dict(state)
    set_training_lr(opt,10001,20000,schedule);set_training_lr(restored,10001,20000,schedule)
    assert opt.param_groups[0]['lr']==restored.param_groups[0]['lr']
    before=opt.param_groups[0]['lr'];set_training_lr(opt,10001,20000,schedule)
    assert opt.param_groups[0]['lr']==before
    print('PASS: ignored/empty/perfect masks, oil gradient, microbatch loss+gradient equivalence, LR endpoints, resume and retry')


if __name__=='__main__':main()
