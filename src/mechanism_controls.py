"""Matched controls for content-dependent weights and detail-branch components."""
import torch
from torch import nn
from torch.nn import functional as F

from .innovations import SRPAInnovations, SpectralDetail
from .model import band_weights


class StaticBandWeights(nn.Module):
    """One learned score per band, shared across every tile (224 parameters)."""
    def __init__(self, bands):
        super().__init__()
        # Zero construction consumes no RNG, preserving subsequent batch sampling.
        # Module names retain compatibility with the existing smoke diagnostics.
        self.score = nn.ParameterDict({'logits': nn.Parameter(torch.zeros(bands))})
        self.quality = nn.Identity()

    def forward(self, x, wavelengths=None, fwhm=None):
        scores = self.score['logits'].unsqueeze(0).expand(x.shape[0], -1)
        return band_weights(scores), None, scores


class ControlledSpectralDetail(SpectralDetail):
    """Retain existing tensors and initialization; bypass one operation at a time."""
    use_semantic_gate = True
    use_high_frequency = True

    def forward(self, x, weights, decoded):
        if self.use_semantic_gate and self.use_high_frequency:
            return super().forward(x, weights, decoded)
        detail = F.gelu(self.norm(self.projection(x * weights[:, :, None, None])))
        if self.use_high_frequency:
            smooth = F.avg_pool2d(F.pad(detail, (1, 1, 1, 1), mode='reflect'), 3, 1)
            detail = detail + F.gelu(self.filter(detail - smooth))
        if self.use_semantic_gate:
            gate = F.interpolate(self.gate(decoded), size=x.shape[-2:],
                                 mode='bilinear', align_corners=False)
            detail = detail * torch.sigmoid(gate)
        return self.output(detail)


class SRPAMechanismControls(SRPAInnovations):
    def __init__(self, *args, reliability_mode='dynamic', detail_use_gate=True,
                 detail_use_high_frequency=True, **kwargs):
        if reliability_mode not in {'dynamic', 'static'}:
            raise ValueError('reliability_mode must be dynamic or static')
        if kwargs.get('local_reliability', False) or kwargs.get('spatial_adapter', False):
            raise ValueError('Mechanism controls require global weights without spatial adapters')
        if not kwargs.get('use_reliability', True) or not kwargs.get('spectral_detail', False):
            raise ValueError('These controls are defined relative to the complete A+B model')
        super().__init__(*args, **kwargs)
        self.reliability_mode = reliability_mode
        if reliability_mode == 'static':
            # Construct the original network first so shared initialization and
            # CPU RNG exactly match the historical dynamic-weight reference.
            self.reliability = StaticBandWeights(len(self.wavelength_nm))
        self.spectral_detail.__class__ = ControlledSpectralDetail
        self.spectral_detail.use_semantic_gate = detail_use_gate
        self.spectral_detail.use_high_frequency = detail_use_high_frequency
        self.spectral_detail.gate.requires_grad_(self.use_spectral_detail and detail_use_gate)
        self.spectral_detail.filter.requires_grad_(self.use_spectral_detail and detail_use_high_frequency)

    def parameter_report(self):
        report = super().parameter_report()
        report['mechanism_control'] = dict(
            reliability_mode=self.reliability_mode,
            semantic_gate=self.spectral_detail.use_semantic_gate,
            high_frequency_residual=self.spectral_detail.use_high_frequency,
            static_weight_parameters=224 if self.reliability_mode == 'static' else 0,
            disabled_tensors_retained_for_matched_initialization=True)
        return report
