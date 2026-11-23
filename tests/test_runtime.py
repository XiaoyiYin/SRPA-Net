"""Ghost BN preserves sequential microbatch outputs, gradients and buffers."""
import copy
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import GhostBatchNorm2d


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for momentum in [.1,None]:
        reference=torch.nn.BatchNorm2d(5,momentum=momentum).double().train()
        ghost=copy.deepcopy(reference)
        ghost.__class__=GhostBatchNorm2d
        ghost.ghost_batch_size=4
        for batch in [8,8,6,4,2]:
            x=torch.randn(batch,5,7,9,dtype=torch.double,requires_grad=True)
            y=x.detach().clone().requires_grad_()
            reference.zero_grad();ghost.zero_grad()
            expected=torch.cat([reference(part) for part in x.split(4)])
            actual=ghost(y)
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
            expected.square().sum().backward();actual.square().sum().backward()
            torch.testing.assert_close(x.grad,y.grad)
            torch.testing.assert_close(reference.weight.grad,ghost.weight.grad)
            torch.testing.assert_close(reference.bias.grad,ghost.bias.grad)
            for key,value in reference.state_dict().items():
                torch.testing.assert_close(value,ghost.state_dict()[key],rtol=0,atol=0)
        reference.eval();ghost.eval()
        torch.testing.assert_close(reference(x),ghost(x),rtol=0,atol=0)
    print('PASS: ghost batch4 matches sequential microbatch4 forward/backward/running statistics, including short batches and eval')


if __name__=='__main__': main()
