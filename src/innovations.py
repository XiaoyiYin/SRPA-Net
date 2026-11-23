"""Exploratory local reliability, spatial adaptation and spectral detail residuals.

These are proposed combinations for this experiment, not claims of priority.
The pretrained SpecAware tokenizer and ViT remain frozen.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import SRPANet, RangeSafeConv2d, band_weights


class LocalReliability(nn.Module):
    """A bounded patch-specific score correction around the global band score."""
    def __init__(self, quality_dim=32, hidden=16, correction_bound=0.5):
        super().__init__()
        self.quality_projection = nn.Linear(quality_dim, hidden)
        self.local_projection = nn.Linear(3, hidden)
        self.score = nn.Linear(hidden, 1)
        self.correction_bound = correction_bound
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x, quality, global_scores, patch_size):
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            mean = F.avg_pool2d(x, patch_size, patch_size)
            variance = (F.avg_pool2d(x.square(), patch_size, patch_size)-mean.square()).clamp_min(0)
            smooth = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, 1)
            residual = F.avg_pool2d((x-smooth).abs(), patch_size, patch_size)
            local = torch.stack((mean, (variance+1e-12).sqrt(), residual), -1)
            local = local.sign()*torch.log1p(local.abs())
            # B,C,Hg,Wg,3 -> B,N,C,3, matching F.unfold row-major order.
            local = local.flatten(2, 3).transpose(1, 2)
        hidden = self.local_projection(local)+self.quality_projection(quality)[:, None]
        delta = self.correction_bound*torch.tanh(self.score(F.gelu(hidden)).squeeze(-1).float())
        return band_weights(global_scores[:, None].float()+delta), delta


class SpatialAdapter(nn.Module):
    """Full-token residual with reliability-conditioned 3x3 and dilated 3x3 paths."""
    def __init__(self, dim=768, rank=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.local = nn.Conv2d(rank, rank, 3, padding=1, groups=rank)
        self.context = nn.Conv2d(rank, rank, 3, padding=2, dilation=2, groups=rank)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens, routing):
        hidden = F.gelu(self.down(self.norm(tokens)))
        grid = math.isqrt(tokens.shape[1]-1)
        assert grid*grid == tokens.shape[1]-1
        spatial = hidden[:, 1:].transpose(1, 2).reshape(tokens.shape[0], -1, grid, grid)
        correction = (self.local(spatial)*routing[:, 0, None, None, None]
                      + self.context(spatial)*routing[:, 1, None, None, None])
        spatial = spatial+F.gelu(correction)
        hidden = torch.cat((hidden[:, :1], spatial.flatten(2).transpose(1, 2)), 1)
        return tokens+self.up(hidden)


class SpectralDetail(nn.Module):
    """All-band, full-resolution decoder skip with a semantic gate.

    This skip does not create a second set of ViT input tokens. It adds a
    zero-initialized logit residual using local detail and decoder context.
    """
    def __init__(self, bands, decoder_dim=64, width=24):
        super().__init__()
        self.projection = RangeSafeConv2d(bands, width, 1, bias=False)
        self.norm = nn.GroupNorm(4, width)
        self.filter = RangeSafeConv2d(width, width, 3, padding=1, groups=width)
        self.gate = RangeSafeConv2d(decoder_dim, 1, 1)
        self.output = RangeSafeConv2d(width, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, weights, decoded):
        detail = F.gelu(self.norm(self.projection(x*weights[:, :, None, None])))
        smooth = F.avg_pool2d(F.pad(detail, (1, 1, 1, 1), mode='reflect'), 3, 1)
        detail = detail+F.gelu(self.filter(detail-smooth))
        gate = self.gate(decoded)
        gate = F.interpolate(gate, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.output(detail*torch.sigmoid(gate))


class SRPAInnovations(SRPANet):
    def __init__(self, *args, local_reliability=False, spatial_adapter=False,
                 spectral_detail=False, use_reliability=True, use_adapters=False,
                 adapter_rank=16, detail_use_reliability=True,
                 reliability_strength=1.0, detail_detach_weights=False,
                 detail_detach_context=False, **kwargs):
        assert not use_adapters
        if not use_reliability and (local_reliability or spatial_adapter):
            raise ValueError('Local reliability/spatial routing require the reliability descriptor; disable both for the factorial controls')
        super().__init__(*args, use_reliability=use_reliability, use_adapters=False,
                         adapter_rank=adapter_rank, **kwargs)
        self.use_local_reliability = local_reliability
        self.use_spatial_adapter = spatial_adapter
        self.use_spectral_detail = spectral_detail
        self.detail_use_reliability = detail_use_reliability
        if not 0 <= reliability_strength <= 1:
            raise ValueError('reliability_strength must be in [0, 1]')
        if local_reliability and reliability_strength != 1:
            raise ValueError('Strength ablation is defined for global reliability only')
        self.reliability_strength = reliability_strength
        self.reliability_schedule_scale = 1.0
        self.detail_detach_weights = detail_detach_weights
        self.detail_detach_context = detail_detach_context
        # Instantiate all new modules in the same order in every arm. Common
        # parameters and RNG state then match, independent of ablation flags.
        self.local_reliability = LocalReliability()
        self.spatial_routers = nn.ModuleList([nn.Linear(64, 2) for _ in range(self.config['depth'])])
        self.spatial_adapters = nn.ModuleList([SpatialAdapter(self.config['embed_dim'], adapter_rank)
                                              for _ in range(self.config['depth'])])
        self.spectral_detail = SpectralDetail(len(self.wavelength_nm))
        self.local_reliability.requires_grad_(local_reliability)
        self.spatial_routers.requires_grad_(spatial_adapter)
        self.spatial_adapters.requires_grad_(spatial_adapter)
        self.condition.requires_grad_(spatial_adapter)
        self.spectral_detail.requires_grad_(spectral_detail)

    @staticmethod
    def aggregate(patches, u, v, weights):
        intermediate = torch.einsum('bncp,bcrp->bncr', patches, v)
        if weights.ndim == 2:
            intermediate = intermediate*weights[:, None, :, None]
        else:
            assert weights.shape == intermediate.shape[:3]
            intermediate = intermediate*weights[..., None]
        return torch.einsum('bncr,bcer->bne', intermediate, u)

    def spatial_routing(self, weights, quality):
        p = weights.float()/weights.shape[-1]
        entropy = -(p*p.clamp_min(1e-12).log()).sum(-1)/math.log(weights.shape[-1])
        summary = torch.stack((weights.std(-1, unbiased=False), weights.min(-1).values,
                               weights.max(-1).values, entropy), -1)
        description = torch.cat((quality.mean(1), (quality*p[..., None]).sum(1), summary), -1)
        condition = self.condition(description)
        return torch.stack([router(condition).float().softmax(-1) for router in self.spatial_routers], 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        target = self.config['img_size']
        assert channels == len(self.wavelength_nm) and max(height, width) <= target
        if self.use_reliability:
            weights, quality, scores = self.reliability(x, self.wavelength_nm, self.fwhm_nm)
            strength = self.reliability_strength*self.reliability_schedule_scale
            if strength != 1:
                weights = 1+strength*(weights-1)
        else:
            # A genuinely disabled factor: do not evaluate the estimator, and
            # use equal weights in BOTH the tokenizer and the detail branch.
            weights = torch.ones(batch, channels, device=x.device, dtype=torch.float32)
            scores = torch.zeros_like(weights)
            quality = None
        if (height, width) != (target, target):
            x = F.pad(x, (0, target-width, 0, target-height), mode='replicate')
        token_weights, local_delta = weights, None
        if self.use_local_reliability:
            token_weights, local_delta = self.local_reliability(x, quality, scores, self.config['patch_size'])
        # Spatial mean of local weights preserves sum-C, and conditions the
        # adapters/detail skip on the effective spectral contribution.
        summary_weights = token_weights.mean(1) if token_weights.ndim == 3 else weights
        routes = self.spatial_routing(summary_weights, quality) if self.use_spatial_adapter else None
        tokens = self.tokenize(x, token_weights)
        grid = target//self.config['patch_size']
        pos = self.position_encoding(self.config['embed_dim'], grid, self.gsd.expand(batch), cls_token=True).to(tokens)
        tokens = tokens+pos[:, 1:]
        cls = self.network.encoder.cls_token+pos[:, :1]
        tokens = torch.cat((cls.expand(batch, -1, -1), tokens), 1)
        features = []
        for layer, block in enumerate(self.network.encoder.blocks):
            if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if self.use_spatial_adapter:
                tokens = self.spatial_adapters[layer](tokens, routes[:, layer])
            if layer == self.config['depth']-1:
                tokens = self.network.encoder.norm(tokens)
            if layer in (2, 5, 8, 11):
                features.append(tokens[:, 1:].transpose(1, 2).reshape(batch, self.config['embed_dim'], grid, grid))
        decoded = self.network.decoder(*self.network.neck(features))
        output = self.network.segmentation_head(decoded)
        if self.use_spectral_detail:
            # Keep reliability in the tokenizer while optionally feeding the
            # original normalized spectrum to the detail skip. The semantic
            # gate still receives decoder context from the weighted backbone.
            detail_weights = summary_weights if self.detail_use_reliability else torch.ones_like(summary_weights)
            if self.detail_detach_weights:
                detail_weights = detail_weights.detach()
            detail_context = decoded.detach() if self.detail_detach_context else decoded
            output = output+self.spectral_detail(x, detail_weights, detail_context)
        self.last_diagnostics = dict(weights=summary_weights.detach(), scores=scores.detach(),
            routing=None if routes is None else routes.detach(),
            local_weights=None if token_weights.ndim == 2 else token_weights.detach(),
            local_delta=None if local_delta is None else local_delta.detach())
        return output[:, :, :height, :width]

    def innovation_groups(self):
        return dict(local_reliability=self.local_reliability, spatial_routers=self.spatial_routers,
                    spatial_adapters=self.spatial_adapters, spectral_detail=self.spectral_detail)

    def parameter_report(self):
        report = super().parameter_report()
        report['groups'].update({name: sum(p.numel() for p in module.parameters() if p.requires_grad)
                                for name, module in self.innovation_groups().items()})
        return report
