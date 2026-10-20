"""Read-only CPU inference for the ten frozen TRPO learners."""
from pathlib import Path
import hashlib
import json

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
STEPS = 4194304
LEARNER_SEED = 1
REGISTRY = Path(__file__).with_name('v3_checkpoints.json')
REGISTRY_SHA256 = 'd3eec9fd8066f02a193dd25238bcbfff3f449c2cf23de4c632828cca0355f4a1'
CONTRACT_SHA256 = {
    0: 'babd6b81d41e965d1a42affb743cc19ed4af6ba5eb8a0752b8e2eb8c0b0b7397',
    1: '00333fe38c5ecc9b365e619a92901b561970b2e92366de311b07d618adbfaa3a',
    2: 'c97b28553db470c083372cd99b49545a03b2840100d5d297443c32bdeea7eeca',
    3: '441ffed4d8faae2c2587d9ef3c191902c749e5d599f39b85acc084e094f9277f',
    4: '66e0b7e5b828b98c446ff27a71b2b1d364befd1188536f5c8a0d747aebc7a8b8',
    5: '16870661d10ac030f5d90a00935006714247f86323e34d7f6fad8decc58ce410',
    6: 'cb3bae4cdcb4cb4761dae72afe3364dec4a6633ac0cc069eee8700d7d3d1eb05',
    7: '5f2f1ff0435d3c66a1e1751e1e34001ba76345700ab5300c0b4d693eea5825d9',
    8: '473b0e3c724c37af8b16376961551e1e022c3596de52f5b6912f6977511bd556',
    9: 'bcb2fcc52bfbbfe736c303387abf797cb0ecab6b8a3765a2be72758dfd2dd571',
}
# Retain the selected T-1 identities for identity checks.
CHECKSUMS = {
    'model.zip': '277567ab4491aa3dda8b55d2b52545edc487761573f532103596d2fc0385f876',
    'manifest.json': 'fc57d9860e0e538e8c73014881aa5a8c445d9c4661bd270b001dc6e24cfc33cd',
    'contract.json': '00333fe38c5ecc9b365e619a92901b561970b2e92366de311b07d618adbfaa3a',
}
CANDIDATES = ('artifacts/trpo/selected-T1', 'runs/study-v3/T-1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_registry():
    if digest(REGISTRY) != REGISTRY_SHA256:
        raise ValueError('The frozen TRPO checkpoint registry differs.')
    registry = json.loads(REGISTRY.read_text())
    targets = {model['learner_seed']: model for model in registry['models']}
    if set(targets) != set(range(10)) or len(registry['models']) != 10:
        raise ValueError('Expected the ten frozen TRPO learners.')
    return targets


def selected_checkpoint(root=ROOT, learner_seed=LEARNER_SEED):
    targets = read_registry()
    if learner_seed not in targets:
        raise ValueError('Choose a TRPO learner seed from 0 to 9.')
    target = targets[learner_seed]
    candidates = CANDIDATES if learner_seed == LEARNER_SEED else (f'runs/study-v3/T-{learner_seed}',)
    for name in candidates:
        folder = Path(root) / name
        if all((folder / file).is_file() for file in CHECKSUMS):
            return folder / 'model.zip', {
                'model_sha256': target['sha256'], 'actual_steps': STEPS,
                'training_contract_sha256': CONTRACT_SHA256[learner_seed],
                'settings': {'condition': 'T', 'seed': learner_seed, 'mode': 'full'},
                'application_version': 'TRPO', 'algorithm': 'MaskableTRPO',
            }
    return None, {}


def application_catalog(root=ROOT):
    """Discover final learners without loading or retaining policy weights."""
    result = []
    for seed in range(10):
        path, details = selected_checkpoint(root, seed)
        if path is not None:
            result.append((path, details))
    return result


def validate_checkpoint(path, root=ROOT):
    """Check exact weights, records and the approved inference source."""
    path, root = Path(path), Path(root)
    if path.name != 'model.zip':
        raise ValueError('Expected a frozen TRPO model.zip.')
    manifest = json.loads(path.with_name('manifest.json').read_text())
    learner_seed = manifest.get('settings', {}).get('seed')
    target = read_registry().get(learner_seed)
    if target is None:
        raise ValueError('Unexpected TRPO learner seed.')
    checksums = {
        'model.zip': target['sha256'],
        'manifest.json': target['manifest_sha256'],
        'contract.json': CONTRACT_SHA256[learner_seed],
    }
    for name, expected in checksums.items():
        if digest(path.with_name(name)) != expected:
            raise ValueError('Frozen TRPO file checksum differs: ' + name)
    contract = json.loads(path.with_name('contract.json').read_text())
    if (manifest['status'] != 'completed' or manifest['actual_steps'] != STEPS
            or manifest['settings']['seed'] != learner_seed
            or manifest['settings']['condition'] != 'T'):
        raise ValueError('Unexpected TRPO learner identity or transition count.')
    from env import config as C
    config = json.loads(json.dumps({k: v for k, v in vars(C).items() if k.isupper()}))
    if manifest['config'] != config:
        raise ValueError('TRPO was trained with a different game configuration.')
    required = {n: h for n, h in manifest['source_hashes'].items()
                if n.startswith(('env/', 'agents/')) or n in ('train/features.py', 'experiments/rollout.py')}
    required['trpo/maskable_trpo.py'] = contract['source_hashes']['maskable_trpo.py']
    for name, expected in required.items():
        actual = digest(root / name)
        if actual != expected:
            from provenance import source_hash_matches
            if not source_hash_matches(name, expected, actual):
                raise ValueError('Frozen TRPO inference source differs: ' + name)
    return manifest


@st.cache_resource(max_entries=2, show_spinner=False)
def _load_cached(path, modified_ns, size_bytes, learner_seed, expected_sha256):
    from trpo.maskable_trpo import MaskableTRPO
    from ui.deployment import CpuPolicy
    from ui.policy_playback import isolated_rng
    with isolated_rng(0):
        model = MaskableTRPO.load(path, device='cpu')
        if model.num_timesteps != STEPS or model.seed != learner_seed:
            raise ValueError('TRPO checkpoint counter or learner seed differs.')
        model.policy.set_training_mode(False)
        return CpuPolicy(model)


def load_selected_policy(path):
    # Recheck files before consulting the cache, including source and sidecars.
    manifest = validate_checkpoint(path)
    stamp = Path(path).stat()
    return _load_cached(str(Path(path).resolve()), stamp.st_mtime_ns, stamp.st_size,
                        manifest['settings']['seed'], digest(path))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check'])
    parser.add_argument('--seed', type=int, default=LEARNER_SEED, choices=range(10))
    args = parser.parse_args()
    path, _ = selected_checkpoint(learner_seed=args.seed)
    if path is None:
        raise SystemExit(f'T-{args.seed} checkpoint files are missing.')
    load_selected_policy(path)
    print(f'TRPO ready: T-{args.seed}, 4,194,304 transitions, CPU inference. Default acting mode: SS.')


if __name__ == '__main__':
    main()
