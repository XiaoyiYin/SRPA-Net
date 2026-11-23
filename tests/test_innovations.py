"""Mathematical and gradient checks for the exploratory additions."""
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.innovations import LocalReliability, SpatialAdapter, SpectralDetail, SRPAInnovations
from src.model import band_weights


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    local = LocalReliability()
    x = torch.rand(2, 7, 8, 8)
    quality = torch.randn(2, 7, 32, requires_grad=True)
    scores = torch.randn(2, 7, requires_grad=True)
    weights, delta = local(x, quality, scores, 4)
    assert weights.shape == (2, 4, 7)
    torch.testing.assert_close(weights, band_weights(scores)[:, None].expand(-1, 4, -1))
    torch.testing.assert_close(weights.sum(-1), torch.full((2, 4), 7.))
    assert delta.count_nonzero() == 0
    nn = torch.nn
    nn.init.normal_(local.score.weight, std=.1)
    weights, delta = local(x, quality, scores, 4)
    assert delta.abs().max() <= .5 and weights.min() > 0
    assert weights.std(1).sum() > 0
    (weights*torch.arange(7)).square().mean().backward()
    for tensor in (scores.grad, quality.grad, local.local_projection.weight.grad):
        assert torch.isfinite(tensor).all() and tensor.abs().sum() > 0
    # Compare explicit per-band contributions and gradients at every token.
    patches = torch.randn(2, 4, 7, 3, dtype=torch.double)
    u = torch.randn(2, 7, 5, 2, dtype=torch.double)
    v = torch.randn(2, 7, 2, 3, dtype=torch.double)
    w = torch.rand(2, 4, 7, dtype=torch.double, requires_grad=True)
    explicit = (torch.einsum('bncp,bcrp,bcer->bnce', patches, v, u)*w[..., None]).sum(2)
    efficient = SRPAInnovations.aggregate(patches, u, v, w)
    torch.testing.assert_close(efficient, explicit)
    a = torch.autograd.grad(explicit.square().sum(), w, retain_graph=True)[0]
    b = torch.autograd.grad(efficient.square().sum(), w)[0]
    torch.testing.assert_close(a, b)
    # Verify local descriptors follow the same row-major patch order as unfold.
    ordered = LocalReliability()
    with torch.no_grad():
        for p in ordered.parameters(): p.zero_()
        ordered.local_projection.weight[0, 0] = 1
        ordered.score.weight[0, 0] = 1
    cube = torch.zeros(1, 2, 8, 8)
    cube[0, 0, :4, :4] = 1
    cube[0, 0, :4, 4:] = 2
    cube[0, 0, 4:, :4] = 3
    cube[0, 0, 4:, 4:] = 4
    w, _ = ordered(cube, torch.zeros(1, 2, 32), torch.zeros(1, 2), 4)
    assert torch.all(w[0, 1:, 0] > w[0, :-1, 0])
    adapter = SpatialAdapter(dim=12, rank=4)
    tokens = torch.randn(2, 10, 12, requires_grad=True)
    logits = torch.randn(2, 2, requires_grad=True)
    routing = logits.softmax(-1)
    torch.testing.assert_close(adapter(tokens, routing), tokens, rtol=0, atol=0)
    nn.init.normal_(adapter.up.weight, std=.1)
    adapter(tokens, routing).square().mean().backward()
    for grad in (logits.grad, adapter.local.weight.grad, adapter.context.weight.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    # A changed token must influence a spatial neighbor through the convolution.
    with torch.no_grad():
        baseline = adapter(tokens, routing)
        changed = tokens.detach().clone()
        changed[:, 5, 0] += 2
        difference = adapter(changed, routing)-baseline
        assert difference[:, 2].abs().sum() > 0
        torch.testing.assert_close(difference[:, 0], torch.zeros_like(difference[:, 0]))
    detail = SpectralDetail(bands=7)
    full = torch.rand(2, 7, 8, 8, requires_grad=True)
    decoded = torch.rand(2, 64, 4, 4, requires_grad=True)
    torch.testing.assert_close(detail(full, torch.ones(2, 7), decoded), torch.zeros(2, 2, 8, 8))
    nn.init.normal_(detail.output.weight, std=.1)
    detail(full, torch.ones(2, 7), decoded).square().mean().backward()
    assert (full.grad.abs().sum((0, 2, 3)) > 0).all()
    assert detail.gate.weight.grad.abs().sum() > 0
    assert torch.isfinite(full.grad).all()
    print('PASS: local sum-C/bounded correction/token order, exact aggregation and gradients; spatial identity/neighbor exchange/routes; all-band detail and semantic gate gradients')


if __name__ == '__main__':
    main()
