"""Real PPO, DDQN and TRPO integration through one application session."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random

import numpy as np
import torch
from streamlit.testing.v1 import AppTest

from env.generator import generate_base
from experiments.rollout import load_replay, save_replay
from ui import ddqn_models, trpo_models
from ui.custom_bases import layout_from_spec
from ui.deployment import load_cpu_policy
from ui.policy_playback import html_bytes, record_policy
from ui.v2_models import read_registry

ROOT = Path(__file__).resolve().parents[1]
POLICIES = (
    ('Saved Maskable PPO', 'MaskablePPO'),
    ('Saved Masked DDQN', 'masked-double-dqn'),
    ('Saved Maskable TRPO', 'MaskableTRPO'),
)


def test_all_three_real_policies_generated_custom_and_exports(tmp_path):
    app = AppTest.from_file(str(ROOT / 'ui/app.py')).run(timeout=30)

    def select(label, value):
        next(widget for widget in app.selectbox if widget.label == label).set_value(value).run(timeout=30)
        assert not app.exception

    select('Mode', 'Policy replay')
    for custom in (False, True):
        if custom:
            app.session_state['custom_layout'] = layout_from_spec(
                generate_base(7, 'closed_ring'), name='Three-method integration')
            select('Play on', 'My custom base')
        for name, algorithm in POLICIES:
            select('Policy', name)
            if algorithm == 'MaskableTRPO':
                select('TRPO learner seed', 9)
            next(button for button in app.button if button.label == 'Generate replay').click().run(timeout=30)
            assert not app.exception
            replay = app.session_state['replay']
            assert app.session_state['replay_policy'] == name
            if algorithm == 'MaskablePPO':
                assert replay['policy']['model_version'] == 'V2'
            else:
                assert replay['policy']['algorithm'] == algorithm
            assert replay['visual_frames'] and replay['actions']
            assert len(app.get('download_button')) == 2
            if custom:
                assert replay['summary']['motif'] == 'custom'
            file = tmp_path / f'{algorithm}-{custom}.json'
            save_replay(replay, file)
            assert load_replay(file)['actions'] == replay['actions']
            assert b'streamlit:render' in html_bytes(replay)


def test_concurrent_real_algorithms_match_sequential_and_restore_rng():
    target = read_registry()['models'][0]
    path = ROOT / target['path']
    stamp = path.stat()
    ppo = load_cpu_policy(str(path), stamp.st_mtime_ns, stamp.st_size,
                          target['sha256'], target['transitions'])
    ddqn = ddqn_models.load_selected_policy(ddqn_models.selected_checkpoint(ROOT)[0])
    trpo = trpo_models.load_selected_policy(trpo_models.selected_checkpoint(ROOT, 9)[0])
    jobs = [(ppo, 'DS', 'v2-final-ds-v1', 41001),
            (ddqn, 'DD', 'legacy', 0),
            (trpo, 'SS', 'v3-final-ds-v1', 41002),
            (ppo, 'DS', 'v2-final-ds-v1', 41001),
            (ddqn, 'DD', 'legacy', 0)]

    def replay(job):
        policy, mode, scheme, seed = job
        result = record_policy(policy, base_seed=42, motif='closed_ring',
                               action_seed=seed, acting_mode=mode, seed_scheme=scheme)
        return result['actions'], result['summary']

    expected = [replay(job) for job in jobs]
    random.seed(123); np.random.seed(456); torch.manual_seed(789)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with ThreadPoolExecutor(max_workers=5) as executor:
        actual = list(executor.map(replay, jobs))
    assert actual == expected
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
