"""Verify curriculum transitions, default identity, retry and resumed gradients."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiment_controls import training_tversky_fn_weight,oil_tversky_loss_sum,oil_dice_loss_sum


def main():
    torch.set_num_threads(2)
    schedule=dict(name='linear_fn_weight',start_update=8000,end_update=14000,
                  start_fn_weight=.5,end_fn_weight=.75)
    for step,expected in [(1,.5),(8000,.5),(11000,.625),(14000,.75),(20000,.75)]:
        assert training_tversky_fn_weight(step,.5,schedule)==expected
    for beta in [.5,.6,.7,.75,.8]:
        assert training_tversky_fn_weight(14000,beta)==beta
    torch.manual_seed(19)
    labels=torch.tensor([[[1,0,255],[0,1,0]],[[0,1,0],[255,0,0]]])
    initial=torch.randn(2,2,2,3)
    assert torch.equal(oil_tversky_loss_sum(initial,labels,training_tversky_fn_weight(8000,.5,schedule)),
                       oil_dice_loss_sum(initial,labels))
    stored=json.loads(json.dumps(dict(step=10000,schedule=schedule)))
    # Resuming at update 10001 must reproduce the uninterrupted objective and gradient.
    beta=training_tversky_fn_weight(stored['step']+1,.5,stored['schedule'])
    full=initial.clone().requires_grad_()
    value=oil_tversky_loss_sum(full,labels,training_tversky_fn_weight(10001,.5,schedule));value.backward()
    sliced=initial.clone().requires_grad_()
    parts=sum(oil_tversky_loss_sum(x,y,beta) for x,y in zip(sliced.split(1),labels.split(1)));parts.backward()
    torch.testing.assert_close(value,parts);torch.testing.assert_close(full.grad,sliced.grad)
    assert training_tversky_fn_weight(10001,.5,schedule)==beta # Retry uses same update.
    assert not (sliced.grad.permute(0,2,3,1)[labels==255]).any()
    for invalid in [dict(schedule,end_update=8000),dict(schedule,start_fn_weight=0),dict(schedule,name='other')]:
        try:training_tversky_fn_weight(1,.5,invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid curriculum accepted')
    print('PASS: Dice identity, curriculum endpoints, resumed/microbatched gradients, retry and invalid schedules')


if __name__=='__main__':main()
