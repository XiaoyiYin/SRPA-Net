"""Execution-only controls; optimizer batch and dataset protocol stay explicit."""
import torch
from torch import nn

from .model import RangeSafeConv2d


class GhostBatchNorm2d(nn.BatchNorm2d):
    """Preserve sequential microbatch BN statistics in a larger physical batch."""
    ghost_batch_size = 4

    def forward(self, x):
        if self.training and x.shape[0] > self.ghost_batch_size:
            return torch.cat([super(GhostBatchNorm2d, self).forward(part)
                              for part in x.split(self.ghost_batch_size)], dim=0)
        return super().forward(x)


def apply_runtime(model, options):
    if not options:
        return
    if 'activation_checkpointing' in options:
        model.activation_checkpointing = options['activation_checkpointing']
    model.fp32_retry_micro_batch=options.get('fp32_retry_micro_batch',options.get('ghost_batch_size',4))
    for module in model.modules():
        if isinstance(module, RangeSafeConv2d):
            # The training/evaluation loop still checks loss/logits and reruns
            # the same batch in FP32, restoring BN buffers and RNG on retries.
            module.defer_range_check = options.get('defer_conv_checks', False)
        if isinstance(module, nn.BatchNorm2d) and options.get('ghost_batch_size'):
            module.__class__ = GhostBatchNorm2d
            module.ghost_batch_size = options['ghost_batch_size']


def set_optimizer_runtime(optimizer, options):
    if options.get('fused_adamw'):
        assert getattr(optimizer, '_step_supports_amp_scaling', False), 'Construct AdamW with fused=True before restoring its state'
        for group in optimizer.param_groups:
            group['fused'] = True
            group['foreach'] = False
            # Old non-fused AdamW checkpoints store scalar steps on the CPU.
            for parameter in group['params']:
                state=optimizer.state.get(parameter,{})
                if 'step' in state:
                    state['step']=state['step'].to(parameter.device)
