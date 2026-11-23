"""SpecAware single tokenizer + SpectralX-inspired full-space adapter mixtures."""
from functools import partial
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]


def read_pretrained(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    for container in ("state_dict", "model", "encoder"):
        if container in state and isinstance(state[container], dict):
            state = state[container]
    cleaned = {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        for prefix in ("module.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = tensor
    return cleaned


def checkpoint_config(state):
    dim = state["cls_token"].shape[-1]
    indices = {int(k.split(".")[1]) for k in state if k.startswith("blocks.")}
    depth = max(indices) + 1
    v_weight = state["patch_embed.matrix_generator.v_generator.6.weight"]
    u_weight = state["patch_embed.matrix_generator.u_generator.6.weight"]
    rank = u_weight.shape[0] // dim
    patch_size = math.isqrt(v_weight.shape[0] // rank)
    patch_count = state["patch_embed.content_feature_extractor.avg_feature_encoder.0.weight"].shape[1]
    grid = math.isqrt(patch_count)
    assert grid * grid == patch_count
    assert patch_size * patch_size * rank == v_weight.shape[0]
    assert dim == 768 and depth == 12, "This implementation expects the published ViT-Base checkpoint."
    return dict(embed_dim=dim, depth=depth, num_heads=12, patch_size=patch_size,
                img_size=grid * patch_size, hyper_rank=rank)


class RangeSafeConv2d(nn.Conv2d):
    def forward(self, x):
        output = super().forward(x)
        if getattr(self, 'defer_range_check', False):
            return output
        if output.dtype == torch.float16 and not torch.isfinite(output).all():
            with torch.autocast("cuda", enabled=False):
                output = super().forward(x.float())
            if not torch.isfinite(output).all():
                raise FloatingPointError("Nonfinite FP32 decoder convolution")
        return output


def band_weights(scores):
    """Sum-C normalization: equal scores give unit weights, not 1/C."""
    return torch.softmax(scores.float(), dim=-1) * scores.shape[-1]


class ReliabilityEstimator(nn.Module):
    """Relative task-learned reliability, never a calibrated physical SNR."""
    def __init__(self, hidden=32):
        super().__init__()
        self.quality = nn.Sequential(nn.Linear(14, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU())
        self.score = nn.Linear(hidden, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    @staticmethod
    def statistics(x, wavelengths, fwhm):
        # Use FP32 for local residual statistics, independent of AMP precision.
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            smooth = F.avg_pool2d(F.pad(x, (1,1,1,1), mode="reflect"), 3, stride=1)
            residual = x - smooth
            mean = x.mean((-2,-1))
            std = x.var((-2,-1), unbiased=False).clamp_min(1e-12).sqrt()
            r_abs = residual.abs().mean((-2,-1))
            r_std = residual.var((-2,-1), unbiased=False).clamp_min(1e-12).sqrt()
            dx = (x[:,:,:,1:] - x[:,:,:,:-1]).abs().mean((-2,-1))
            dy = (x[:,:,1:,:] - x[:,:,:-1,:]).abs().mean((-2,-1))
            pooled = F.adaptive_avg_pool2d(x, (2,2)).flatten(2)
            base = torch.stack((mean, std, r_abs, r_std, r_std/(std+1e-4), dx, dy,
                                x.square().mean((-2,-1))), -1)
            meta = torch.stack((wavelengths/2550., fwhm/15.), -1)
            meta = meta.unsqueeze(0).expand(x.shape[0],-1,-1)
            # The original reader leaves constant nonzero bands unchanged.
            # Signed log scaling prevents their squared-energy descriptors
            # from overflowing an FP16 quality MLP; image values stay intact.
            descriptors=torch.cat((base,pooled),-1)
            descriptors=descriptors.sign()*torch.log1p(descriptors.abs())
            return torch.cat((descriptors, meta), -1)

    def forward(self, x, wavelengths, fwhm):
        quality = self.quality(self.statistics(x, wavelengths, fwhm))
        scores = self.score(quality).squeeze(-1)
        return band_weights(scores), quality, scores


class ConditionalAdapters(nn.Module):
    """Full D-dimensional residual; no assumed spatial/spectral channel halves."""
    def __init__(self, dim=768, rank=16, experts=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim,rank), nn.GELU(), nn.Linear(rank,dim))
            for _ in range(experts)
        ])
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(self, tokens, routing):
        normalized = self.norm(tokens)
        correction = sum(expert(normalized) * routing[:,k,None,None]
                         for k, expert in enumerate(self.experts))
        return tokens + correction


class SRPANet(nn.Module):
    def __init__(self, pretrained_path, metadata, text_embeddings,
                 use_reliability=True, use_adapters=True, adapter_rank=16,
                 experts=4, activation_checkpointing=True):
        super().__init__()
        self.use_reliability = use_reliability
        self.use_adapters = use_adapters
        self.activation_checkpointing = activation_checkpointing
        state = read_pretrained(pretrained_path)
        self.config = checkpoint_config(state)
        sys.path.insert(0, str(ROOT/"vendor/specaware"))
        import SpectralEmbedding as spectral
        cache = torch.load(text_embeddings, map_location="cpu", weights_only=True)

        class CachedSentenceEncoder(nn.Module):
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                super().__init__()
                self.embedding_dim=384
            def forward(self, sentence):
                return cache[sentence].clone()

        spectral.SentenceEncoder = CachedSentenceEncoder
        from models_SpecAware_upernet import ViTUPerNet
        from util.pos_embed import get_2d_sincos_pos_embed_with_resolution
        self.position_encoding = get_2d_sincos_pos_embed_with_resolution
        self.network = ViTUPerNet(
            img_size=self.config["img_size"], in_chans=len(metadata["wavelength_nm"]),
            num_classes=2, embed_dim=self.config["embed_dim"],depth=self.config["depth"],
            num_heads=self.config["num_heads"],patch_size=self.config["patch_size"],
        )
        # Match the actual encoder constructor: the upstream UPerNet wrapper
        # uses eps=1e-6 in its encoder, and strict loading checks every tensor.
        encoder = self.network.encoder
        expected = encoder.state_dict()
        missing = sorted(set(expected)-set(state))
        mismatched = {k:(tuple(state[k].shape),tuple(v.shape)) for k,v in expected.items()
                      if k in state and state[k].shape != v.shape}
        if missing or mismatched:
            raise ValueError(f"Pretrained encoder incompatible: missing={missing}; shapes={mismatched}")
        encoder.load_state_dict({k:state[k] for k in expected}, strict=True)
        self.pretrained_loading = dict(loaded_encoder_tensors=len(expected),
            ignored_pretraining_only_tensors=sorted(set(state)-set(expected)), strict=True)
        encoder.requires_grad_(False)
        encoder.eval()
        encoder.patch_embed.spectral_embedding.attribute_encoder.processed_cache["unknown"] = cache["Unknown processing level"].clone()
        for name,module in self.network.named_modules():
            if name.startswith(("neck.","decoder.")) and isinstance(module,nn.Conv2d):
                module.__class__=RangeSafeConv2d
        self.reliability = ReliabilityEstimator()
        self.condition = nn.Sequential(nn.Linear(68,64),nn.GELU())
        self.routers = nn.ModuleList([nn.Linear(64,experts) for _ in range(self.config["depth"])])
        self.adapters = nn.ModuleList([ConditionalAdapters(self.config["embed_dim"],adapter_rank,experts)
                                       for _ in range(self.config["depth"])])
        if not use_reliability:
            self.reliability.score.requires_grad_(False)
        if not use_adapters:
            self.condition.requires_grad_(False)
            self.routers.requires_grad_(False)
            self.adapters.requires_grad_(False)
            if not use_reliability:
                self.reliability.requires_grad_(False)
        self.register_buffer("wavelength_nm",torch.tensor(metadata["wavelength_nm"],dtype=torch.float32))
        self.register_buffer("fwhm_nm",torch.tensor(metadata["fwhm_nm"],dtype=torch.float32))
        self.register_buffer("gsd",torch.tensor(float(metadata["GSD"])))
        self.sensor_name=metadata["sensor_name"]
        self.data_level=metadata["processing_level"]
        self.last_diagnostics={}

    def train(self,mode=True):
        super().train(mode)
        self.network.encoder.eval()  # Frozen dropout/state must stay deterministic.
        return self

    @staticmethod
    def aggregate(patches, u, v, weights):
        # Algebraically sum_c w_c * contribution_c, without allocating B*N*C*D.
        intermediate=torch.einsum("bncp,bcrp->bncr",patches,v)
        intermediate=intermediate * weights[:,None,:,None]
        return torch.einsum("bncr,bcer->bne",intermediate,u)

    def tokenize(self,x,weights):
        embedding=self.network.encoder.patch_embed
        batch,channels,height,width=x.shape
        with torch.no_grad():
            meta=embedding.spectral_embedding(self.wavelength_nm.expand(batch,-1),self.fwhm_nm.expand(batch,-1),self.sensor_name,self.data_level)
            meta=embedding.spectral_aware_transformer(meta)
            content=embedding.content_feature_extractor(x)
            fused=embedding.feature_fusion(meta,content)+meta
            u,v,bias=embedding.matrix_generator(fused)
            ps=embedding.kernel_size
            patches=F.unfold(x,kernel_size=ps,stride=ps).transpose(1,2)
            patches=patches.reshape(batch,-1,channels,ps*ps)
        # Do not detach weights or wrap the aggregation in no_grad: task
        # gradients must reach reliability through the frozen tokenizer.
        output=self.aggregate(patches,u,v,weights)
        return (output+bias[:,None,:])*embedding.scaler

    def routing(self,weights,quality):
        p=weights.float()/weights.shape[-1]
        weighted=(quality*p[:,:,None]).sum(1)
        entropy=-(p*p.clamp_min(1e-12).log()).sum(-1)/math.log(max(2,weights.shape[-1]))
        summary=torch.stack((weights.std(-1,unbiased=False),weights.min(-1).values,
                             weights.max(-1).values,entropy),-1)
        condition=self.condition(torch.cat((quality.mean(1),weighted,summary),-1))
        return torch.stack([F.softmax(router(condition).float(),-1) for router in self.routers],1)

    def forward(self,x):
        batch,channels,height,width=x.shape
        assert channels==len(self.wavelength_nm)
        assert height<=self.config["img_size"] and width<=self.config["img_size"]
        weights,quality,scores=self.reliability(x,self.wavelength_nm,self.fwhm_nm)
        if not self.use_reliability:
            weights=torch.ones_like(weights)
        routing=self.routing(weights,quality) if self.use_adapters else None
        target=self.config["img_size"]
        if (height,width)!=(target,target):
            x=F.pad(x,(0,target-width,0,target-height),mode="replicate")
        tokens=self.tokenize(x,weights)
        grid=target//self.config["patch_size"]
        pos=self.position_encoding(self.config["embed_dim"],grid,self.gsd.expand(batch),cls_token=True).to(tokens)
        tokens=tokens+pos[:,1:,:]
        cls=self.network.encoder.cls_token+pos[:,:1,:]
        tokens=torch.cat((cls.expand(batch,-1,-1),tokens),1)
        features=[]
        for layer,block in enumerate(self.network.encoder.blocks):
            if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
                tokens=checkpoint(block,tokens,use_reentrant=False)
            else:
                tokens=block(tokens)
            if self.use_adapters:
                tokens=self.adapters[layer](tokens,routing[:,layer,:])
            if layer==self.config["depth"]-1:
                tokens=self.network.encoder.norm(tokens)
            if layer in (2,5,8,11):
                features.append(tokens[:,1:,:].transpose(1,2).reshape(batch,self.config["embed_dim"],grid,grid))
        features=self.network.neck(features)
        output=self.network.segmentation_head(self.network.decoder(*features))
        self.last_diagnostics={"weights":weights.detach(),"scores":scores.detach(),
                               "routing":None if routing is None else routing.detach()}
        return output[:,:,:height,:width]

    def parameter_report(self):
        total=sum(p.numel() for p in self.parameters())
        trainable=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return dict(total=total,trainable=trainable,frozen=total-trainable,trainable_fraction=trainable/total,
                    groups={name:sum(p.numel() for p in module.parameters() if p.requires_grad)
                            for name,module in [("reliability",self.reliability),("condition",self.condition),
                             ("routers",self.routers),("adapters",self.adapters),("neck",self.network.neck),
                             ("decoder",self.network.decoder),("segmentation_head",self.network.segmentation_head)]})
