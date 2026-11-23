"""Weighting-location control with the formal E03 gradient policy unchanged."""
import torch
from torch.nn import functional as F

from .innovations import SRPAInnovations


class SRPAWeightingLocation(SRPAInnovations):
    def __init__(self, *args, weighting_location='raw_input', **kwargs):
        super().__init__(*args, **kwargs)
        assert weighting_location in {'raw_input', 'aggregation'}
        assert self.use_reliability and self.use_spectral_detail
        assert self.detail_detach_weights and not self.detail_detach_context
        assert not self.use_local_reliability and not self.use_spatial_adapter
        self.weighting_location = weighting_location

    def tokenize(self, x, weights):
        if self.weighting_location == 'aggregation':
            return super().tokenize(x, weights)
        assert weights.ndim == 2 and weights.shape == x.shape[:2]
        embedding = self.network.encoder.patch_embed
        batch, channels, _, _ = x.shape
        # Estimate w from the original X in the inherited forward, then feed
        # wX to BOTH the content-conditioned factor generator and patch path.
        weighted_x = x * weights[:, :, None, None]
        with torch.no_grad():
            meta = embedding.spectral_embedding(
                self.wavelength_nm.expand(batch, -1), self.fwhm_nm.expand(batch, -1),
                self.sensor_name, self.data_level)
            meta = embedding.spectral_aware_transformer(meta)
            content = embedding.content_feature_extractor(weighted_x)
            fused = embedding.feature_fusion(meta, content) + meta
            u, v, bias = embedding.matrix_generator(fused)
        # E03 detaches generated factors. Keep that policy, but retain the
        # gradient from weighted patches to w. Never apply w a second time.
        ps = embedding.kernel_size
        patches = F.unfold(weighted_x, kernel_size=ps, stride=ps).transpose(1, 2)
        patches = patches.reshape(batch, -1, channels, ps * ps)
        output = self.aggregate(patches, u, v, torch.ones_like(weights))
        return (output + bias[:, None, :]) * embedding.scaler
