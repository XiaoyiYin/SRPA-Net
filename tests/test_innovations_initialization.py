"""Verify identical initial common weights and predictions across all arms."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import configure, build_model
from src.data import OilSpillTiles


def digest(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith(('network.', 'reliability.', 'condition.')):
            result.update(name.encode())
            result.update(tensor.cpu().numpy().tobytes())
    return result.hexdigest()


def main():
    plan = json.loads((ROOT/'configs/innovations_v2.json').read_text())
    plan['sensor_metadata'] = json.loads(Path(plan['metadata']).read_text())
    old = dict(plan, model_family='original', variants={'reference': dict(use_reliability=True, use_adapters=False)})
    data = OilSpillTiles(plan['cache_root'], 'tr')
    images = torch.stack([data[i][0] for i in range(2)]).cuda()
    reference, expected_digest = None, None
    for variant in ['reference', *plan['variants']]:
        configure(plan['seed'])
        model = build_model(old if variant == 'reference' else plan, variant).cuda().eval()
        value = digest(model)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            output = model(images)
        assert torch.isfinite(output).all()
        if reference is None:
            reference, expected_digest = output.detach().clone(), value
        else:
            assert value == expected_digest, variant
            torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)
        print(json.dumps(dict(variant=variant, common_weights_sha256=value,
            max_initial_logit_difference=float((output-reference).abs().max()),
            trainable_parameters=model.parameter_report()['trainable'])), flush=True)
        del model, output
        gc.collect()
        torch.cuda.empty_cache()
    print('PASS: five arms have identical initial common weights and match original reliability-only predictions')


if __name__ == '__main__':
    main()
