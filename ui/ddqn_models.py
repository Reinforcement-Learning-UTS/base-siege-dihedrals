"""Pinned DDQN CPU inference; scientific training and recovery stay unchanged."""
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import re
import threading

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
STEPS = 2097152
POLICY_SHA256 = 'c76232fc77ca43bf2cab616d2546324d9026d7a870df28f17ba4388b458fbdc9'
CONTRACT_SHA256 = 'da9cbc73ab732fb98f2b9c0ce35b2da58601cf6f789b04f99949d9dcd433bf86'
SIDECAR_SHA256 = '9f26a391ce895d2826b4927a4f1952f95df4668a2166aaa537a174a0e9df9b46'
RUNTIME_PACKAGES = ('numpy', 'torch', 'gymnasium', 'stable-baselines3', 'sb3-contrib')
CANDIDATES = (
    'artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/selected_model',
    'runs/ddqn/pilot-lr3e5-seed100',
)


def selected_checkpoint(root):
    """Prefer the frozen inference copy of the exact selected checkpoint."""
    for name in CANDIDATES:
        run = Path(root)/name
        policy = run/'policies'/f'step-{STEPS:012d}.pt'
        if all(p.is_file() for p in (policy, policy.with_suffix('.json'), run/'contract.json')):
            return policy, {
                'model_sha256': POLICY_SHA256, 'actual_steps': STEPS,
                'settings': {'condition': 'DDQN', 'seed': 100, 'mode': 'pilot'},
                'application_version': 'DDQN', 'algorithm': 'masked-double-dqn',
                'training_contract_sha256': CONTRACT_SHA256,
            }
    return None, {}


class DDQNPolicy:
    """Same predict interface as the existing replay recorder expects."""
    def __init__(self, network):
        self.network = network
        self.rng = np.random.default_rng(0)  # Not drawn from at epsilon zero.
        self.lock = threading.Lock()

    def predict(self, observation, *, action_masks, deterministic=True):
        if not deterministic:
            raise ValueError('Selected DDQN replay uses greedy legal actions only.')
        from ddqn_experiment.core import choose_action
        with self.lock:
            return choose_action(self.network, observation, action_masks, self.rng, 0.0, 'cpu'), None


def inference_runtime():
    """Measure the real process; callers cannot override the loader's runtime."""
    import torch
    return {'python': platform.python_version(),
            'packages': {name: version(name) for name in RUNTIME_PACKAGES},
            'torch_cuda': torch.version.cuda}


def validate_inference_runtime(recorded, actual):
    """Allow final Python 3.12 patch updates and the matching CPU Torch wheel only."""
    if not re.fullmatch(r'3\.12\.(0|[1-9][0-9]*)', actual.get('python', '')):
        raise ValueError('DDQN CPU inference requires a final Python 3.12 patch release.')
    if not re.fullmatch(r'3\.12\.(0|[1-9][0-9]*)', recorded.get('python', '')):
        raise ValueError('Unexpected recorded DDQN Python release.')
    expected = recorded.get('packages', {})
    packages = actual.get('packages', {})
    if set(expected) != set(RUNTIME_PACKAGES) or set(packages) != set(expected):
        raise ValueError('DDQN inference package inventory differs from training.')
    for name, expected_version in expected.items():
        allowed = (expected_version,)
        if name == 'torch':
            if expected_version != '2.13.0':
                raise ValueError('Unexpected recorded DDQN Torch release.')
            allowed = ('2.13.0', '2.13.0+cpu')
        if packages[name] not in allowed:
            raise ValueError('Unsupported DDQN inference package release: ' + name)
    if 'torch_cuda' not in actual or actual['torch_cuda'] is not None:
        raise ValueError('DDQN inference requires a CPU Torch build without a CUDA runtime.')


def validate_checkpoint(policy_path):
    """Verify original files and approved source before consulting shared weights."""
    from ddqn_experiment.runner import sources
    from provenance import source_tables_match
    path = Path(policy_path)
    run = path.parent.parent
    digest = lambda file: hashlib.sha256(Path(file).read_bytes()).hexdigest()
    if path.name != f'step-{STEPS:012d}.pt' or digest(path) != POLICY_SHA256:
        raise ValueError('Selected DDQN policy differs from the verified checkpoint.')
    if digest(run/'contract.json') != CONTRACT_SHA256:
        raise ValueError('Selected DDQN training contract differs.')
    if digest(path.with_suffix('.json')) != SIDECAR_SHA256:
        raise ValueError('Selected DDQN policy sidecar differs.')
    contract = json.loads((run/'contract.json').read_text())
    metadata = json.loads(path.with_suffix('.json').read_text())
    if (contract.get('schema') != 1 or contract.get('algorithm') != 'masked-double-dqn'
            or contract.get('seed') != 100 or contract.get('mode') != 'pilot'
            or contract.get('condition') != 'no-augmentation'):
        raise ValueError('Unexpected DDQN training identity.')
    if (metadata.get('actual_steps') != STEPS or metadata.get('sha256') != POLICY_SHA256
            or metadata.get('contract_sha256') != CONTRACT_SHA256):
        raise ValueError('Unexpected DDQN sidecar identity.')
    if not source_tables_match(contract['sources'], sources()):
        raise ValueError('DDQN inference source differs from the approved training source.')
    actual = inference_runtime()
    validate_inference_runtime(contract['runtime'], actual)
    return actual


def validate_network_state(state, network):
    """Reject training payloads, wrong identities and incompatible tensor layouts."""
    import torch
    if not isinstance(state, dict) or set(state) != {'schema', 'algorithm', 'steps', 'contract_sha256', 'online'}:
        raise ValueError('Expected the DDQN inference-only state schema.')
    if (type(state['schema']) is not int or state['schema'] != 1
            or state['algorithm'] != 'masked-double-dqn'
            or type(state['steps']) is not int or state['steps'] != STEPS
            or state['contract_sha256'] != CONTRACT_SHA256):
        raise ValueError('Unexpected DDQN checkpoint identity or transition count.')
    online = state['online']
    expected = network.state_dict()
    if not isinstance(online, dict) or set(online) != set(expected):
        raise ValueError('DDQN network parameter names differ.')
    for name, wanted in expected.items():
        tensor = online[name]
        if (type(tensor) is not torch.Tensor or tensor.layout != torch.strided
                or tensor.device.type != 'cpu' or tensor.dtype != wanted.dtype
                or tensor.shape != wanted.shape or not bool(torch.isfinite(tensor).all())):
            raise ValueError('DDQN network tensor differs: ' + name)
    network.load_state_dict(online, strict=True)


@st.cache_resource(max_entries=1, show_spinner=False)
def _load_cached(path, modified_ns, size_bytes, policy_sha256, contract_sha256, sidecar_sha256, runtime_key):
    import torch
    from ddqn_experiment.core import QNetwork
    from env.siege_env import SiegeEnv
    from ui.policy_playback import isolated_rng
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != POLICY_SHA256:
        raise ValueError('Selected DDQN policy differs from the verified checkpoint.')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    with isolated_rng(0):
        env = SiegeEnv()
        try:
            network = QNetwork(env.observation_space).to('cpu')
        finally:
            env.close()
        validate_network_state(state, network)
        network.eval()
        network.requires_grad_(False)
    return DDQNPolicy(network)


def load_selected_policy(policy_path):
    actual = validate_checkpoint(policy_path)
    path = Path(policy_path)
    stamp = path.stat()
    return _load_cached(str(path.resolve()), stamp.st_mtime_ns, stamp.st_size,
                        POLICY_SHA256, CONTRACT_SHA256, SIDECAR_SHA256,
                        json.dumps(actual, sort_keys=True))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check'])
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path, _ = selected_checkpoint(root)
    if path is None:
        raise SystemExit('Selected DDQN files not found. Keep the original run or frozen selected_model folder in this project.')
    load_selected_policy(path)
    print(f'DDQN ready: seed 100, {STEPS:,} transitions, greedy legal acting.\n{path.relative_to(root)}')


if __name__ == '__main__':
    main()
