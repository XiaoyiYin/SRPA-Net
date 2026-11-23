"""Optional training controls; defaults preserve the original experiment."""
from datetime import datetime, timezone
import math


def training_lr_scale(update, max_steps, schedule):
    """One-based optimizer-update schedule, independent of resume/retry state."""
    if not schedule:
        return 1.0
    if schedule['name'] != 'warmup_cosine':
        raise ValueError('Unknown LR schedule')
    warmup = schedule['warmup_steps']
    minimum = schedule['min_lr_ratio']
    start = schedule['warmup_start_ratio']
    if not (1 <= update <= max_steps and 1 < warmup < max_steps
            and 0 < minimum <= 1 and 0 < start <= 1):
        raise ValueError('Invalid LR schedule parameters')
    if update <= warmup:
        return start + (1 - start) * (update - 1) / (warmup - 1)
    progress = (update - warmup) / (max_steps - warmup)
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


def set_training_lr(optimizer, update, max_steps, schedule):
    if not schedule:
        return
    scale = training_lr_scale(update, max_steps, schedule)
    for group in optimizer.param_groups:
        base = group.setdefault('schedule_base_lr', group['lr'])
        group['lr'] = base * scale


def oil_dice_loss_sum(logits, labels):
    """Sum of per-image soft oil Dice losses, over oil-positive images only.

    Ignore pixels contribute neither probability nor target. The caller divides
    by the number of oil-positive images in the full effective batch, preserving
    the same loss under microbatching and FP32 retry. CE covers negative images.
    """
    import torch
    valid = labels != 255
    target = (labels == 1).float()
    probability = logits.float().softmax(1)[:, 1] * valid
    intersection = (probability * target).sum((1, 2))
    mass = target.sum((1, 2))
    loss = 1 - (2 * intersection + 1) / (probability.sum((1, 2)) + mass + 1)
    return (loss * (mass > 0)).sum()


def oil_tversky_loss_sum(logits, labels, fn_weight=0.5):
    """Oil-positive per-image overlap loss; beta weights FN, 1-beta weights FP.

    Smoothing 0.5 makes beta=0.5 equivalent to Dice with smoothing 1.
    Dispatching to the original implementation preserves default arithmetic.
    The caller divides by positive-image count across the effective batch.
    """
    if not 0 < fn_weight < 1:
        raise ValueError('Tversky false-negative weight must be in (0, 1)')
    if fn_weight == 0.5:
        return oil_dice_loss_sum(logits, labels)
    valid = labels != 255
    target = (labels == 1).float()
    probability = logits.float().softmax(1)[:, 1] * valid
    tp = (probability * target).sum((1, 2))
    fp = (probability * (1 - target)).sum((1, 2))
    fn = ((1 - probability) * target).sum((1, 2))
    loss = 1 - (tp + 0.5) / (tp + (1 - fn_weight) * fp + fn_weight * fn + 0.5)
    return (loss * (target.sum((1, 2)) > 0)).sum()


def training_tversky_fn_weight(update, default, schedule=None):
    """One-based loss curriculum; deterministic under resume and FP32 retry."""
    if not 0 < default < 1:
        raise ValueError('Invalid default Tversky weight')
    if not schedule:
        return default
    if schedule['name'] != 'linear_fn_weight':
        raise ValueError('Unknown overlap-loss schedule')
    start, end = schedule['start_update'], schedule['end_update']
    low, high = schedule['start_fn_weight'], schedule['end_fn_weight']
    if not (update >= 1 and 1 <= start < end and 0 < low < 1 and 0 < high < 1):
        raise ValueError('Invalid overlap-loss schedule')
    fraction = min(1.0, max(0.0, (update - start) / (end - start)))
    return low + fraction * (high - low)


def deadline_reached(deadline, current=None):
    if deadline is None:
        return False
    stop = datetime.fromisoformat(deadline)
    if stop.tzinfo is None:
        raise ValueError('Deadline must include its timezone')
    return (current or datetime.now(timezone.utc)) >= stop


def reliability_schedule_scale(step, schedule):
    if not schedule:
        return 1.0
    delay = schedule['delay_steps']
    ramp = schedule['ramp_steps']
    if delay < 0 or ramp <= 0:
        raise ValueError('Invalid reliability schedule')
    return min(1.0, max(0.0, (step-delay)/ramp))


def optimizer_parameters(model, lr, reliability_multiplier=1.0):
    if reliability_multiplier == 1:
        return (p for p in model.parameters() if p.requires_grad)
    if reliability_multiplier <= 0:
        raise ValueError('Reliability learning-rate multiplier must be positive')
    reliability = [p for p in model.reliability.parameters() if p.requires_grad]
    ids = {id(p) for p in reliability}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    if not reliability:
        raise ValueError('No trainable reliability parameters')
    return [dict(params=other, lr=lr),
            dict(params=reliability, lr=lr*reliability_multiplier)]
