"""Direct weighting-location comparison under shared-gradient CE training."""
import torch
from torch.nn import functional as F
from .innovations import SRPAInnovations


class SRPASharedWeightingLocation(SRPAInnovations):
    def __init__(self,*args,weighting_location='aggregation',**kwargs):
        super().__init__(*args,**kwargs)
        assert weighting_location in {'aggregation','raw_input'}
        assert self.use_reliability and self.use_spectral_detail and self.detail_use_reliability
        assert not self.detail_detach_weights and not self.detail_detach_context
        assert not self.use_local_reliability and not self.use_spatial_adapter
        assert self.reliability_strength==1
        self.weighting_location=weighting_location

    def tokenize(self,x,weights):
        if self.weighting_location=='aggregation':
            return super().tokenize(x,weights)
        assert weights.ndim==2 and weights.shape==x.shape[:2]
        embedding=self.network.encoder.patch_embed
        batch,channels,_,_=x.shape
        weighted_x=x*weights[:,:,None,None]
        # Metadata is independent of learned w and all encoder parameters are frozen.
        with torch.no_grad():
            meta=embedding.spectral_embedding(self.wavelength_nm.expand(batch,-1),
                self.fwhm_nm.expand(batch,-1),self.sensor_name,self.data_level)
            meta=embedding.spectral_aware_transformer(meta)
        # Freezing parameters must NOT cut dG(wX)/dw. Retain this input gradient.
        content=embedding.content_feature_extractor(weighted_x)
        fused=embedding.feature_fusion(meta,content)+meta
        u,v,bias=embedding.matrix_generator(fused)
        ps=embedding.kernel_size
        patches=F.unfold(weighted_x,kernel_size=ps,stride=ps).transpose(1,2)
        patches=patches.reshape(batch,-1,channels,ps*ps)
        # w already acts on the input; unit aggregation weights prevent w squared.
        output=self.aggregate(patches,u,v,torch.ones_like(weights))
        return (output+bias[:,None,:])*embedding.scaler
