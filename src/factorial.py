"""Descriptive 2x2 effects in percentage points; never a significance test."""
import json


def summarize(run_dir, plan):
    arms = plan['factorial_arms']
    metrics = plan['factorial_analysis']['metrics']
    available = {}
    for arm, variant in arms.items():
        path = run_dir/'models'/variant/'result.json'
        if path.exists():
            result = json.loads(path.read_text())
            if result.get('status') == 'complete':
                available[arm] = dict(variant=variant, checkpoint_step=result['best_step'],
                    test_percent={m: 100*result['test_best_validation'][m] for m in metrics})
    effects = {}
    if len(available) == 4:
        for metric in metrics:
            base, a, b, ab = (available[key]['test_percent'][metric]
                              for key in ['baseline', 'A', 'B', 'AB'])
            effects[metric] = dict(A_minus_baseline=a-base, B_minus_baseline=b-base,
                AB_minus_baseline=ab-base, AB_minus_A=ab-a, AB_minus_B=ab-b,
                interaction_AB_minus_A_minus_B_plus_baseline=ab-a-b+base)
    return dict(status='complete' if len(available)==4 else 'pending',
                arms=available, pending=[key for key in arms if key not in available],
                effects_percentage_points=effects,
                selection='Maximum validation mIoU independently per arm; test metrics after training only.',
                interpretation='Descriptive single-seed effects on bounded metrics; positive interaction alone is not statistical significance.')
