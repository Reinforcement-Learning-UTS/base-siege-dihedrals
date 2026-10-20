"""Bounded inference checks on synthetic and pinned selected DDQN weights."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest
from ddqn_experiment.core import QNetwork
from ddqn_experiment.runner import sources, runtime
from env.generator import generate_base,BaseSpec
from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.provenance import sha256
from experiments.rollout import play_episode, load_replay, save_replay
from ui import ddqn_models as D
from ui.policy_playback import record_policy, html_bytes
ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH=ROOT/'tests/fixtures/ddqn-runtime-reference.json'
REFERENCE=json.loads(REFERENCE_PATH.read_text())
REFERENCE_SHA256='eea63f7ad5c85706be4f2b773b840896cbe94d1ef7d530d9f4699e1f19fe7b20'


@pytest.fixture
def policy():
    torch.set_num_threads(1)
    env=SiegeEnv();net=QNetwork(env.observation_space);env.close()
    with torch.no_grad():
        for p in net.parameters():p.zero_()
    net.eval()
    return D.DDQNPolicy(net)


def copy_selected_checkpoint(destination):
    original, _ = D.selected_checkpoint(ROOT)
    (destination/'policies').mkdir(parents=True)
    target = destination/'policies'/original.name
    shutil.copy2(original, target)
    shutil.copy2(original.with_suffix('.json'), target.with_suffix('.json'))
    shutil.copy2(original.parent.parent/'contract.json', destination/'contract.json')
    return target


def test_selected_model_integrity_and_frozen_preference(tmp_path):
    frozen=tmp_path/D.CANDIDATES[0]
    path=copy_selected_checkpoint(frozen)
    assert D.selected_checkpoint(tmp_path)[0]==path
    loaded=D.load_selected_policy(path)
    assert D.load_selected_policy(path) is loaded
    assert not loaded.network.training
    assert all(not parameter.requires_grad for parameter in loaded.network.parameters())
    assert all(parameter.device.type=='cpu' for parameter in loaded.network.parameters())
    env=SiegeEnv();obs,_=env.reset(seed=42);mask=env.action_masks()
    assert mask[loaded.predict(obs,action_masks=mask)[0]]
    env.close()
    stamp=path.stat()
    changed=bytearray(path.read_bytes());changed[-1]^=1;path.write_bytes(changed)
    os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    with pytest.raises(ValueError,match='differs'):D.load_selected_policy(path)


@pytest.mark.parametrize('target', ['sidecar', 'contract'])
def test_selected_sidecars_remain_exact_and_are_rechecked_before_cache(tmp_path,target):
    path=copy_selected_checkpoint(tmp_path/'selected')
    D.load_selected_policy(path)
    file=path.with_suffix('.json') if target=='sidecar' else path.parent.parent/'contract.json'
    file.write_bytes(file.read_bytes()+b' ')
    with pytest.raises(ValueError,match='differs'):D.load_selected_policy(path)


def recorded_runtime():
    path,_=D.selected_checkpoint(ROOT)
    return json.loads((path.parent.parent/'contract.json').read_text())['runtime']


@pytest.mark.parametrize('python', ['3.12.0','3.12.9','3.12.14','3.12.99'])
@pytest.mark.parametrize('torch_version', ['2.13.0','2.13.0+cpu'])
def test_inference_runtime_profile_allows_only_documented_patch_and_cpu_variation(python,torch_version):
    recorded=recorded_runtime()
    actual=deepcopy(recorded);actual['python']=python;actual['packages']['torch']=torch_version
    actual['torch_cuda']=None
    D.validate_inference_runtime(recorded,actual)


@pytest.mark.parametrize('field,value', [
    ('python','3.11.14'),('python','3.13.0'),('python','3.12.14rc1'),
    ('python','3.12.14.dev1'),('torch','2.13.1'),('torch','2.14.0'),
    ('torch','2.13.0+cu130'),('torch','2.13.0.dev1'),('torch','2.13.0rc1'),
    ('torch','2.13.0+unknown'),('numpy','2.5.3'),('torch_cuda','13.0'),
])
def test_inference_runtime_profile_rejects_untested_releases(field,value):
    recorded=recorded_runtime();actual=deepcopy(recorded);actual['torch_cuda']=None
    if field in ('python','torch_cuda'):actual[field]=value
    else:actual['packages'][field]=value
    with pytest.raises(ValueError):D.validate_inference_runtime(recorded,actual)


def test_original_scientific_loader_keeps_exact_runtime_guard(tmp_path):
    from ddqn_experiment.evaluation import load_policy
    contract={'sources':sources(),'runtime':{**runtime(),'python':'0.0.0'}}
    (tmp_path/'contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='Runtime changed since training'):
        load_policy(tmp_path,steps=D.STEPS,device='cpu')


@pytest.mark.parametrize('change', ['steps','schema','extra','dtype','shape','missing','nonfinite'])
def test_inference_state_rejects_wrong_identity_and_tensor_layout(policy,change):
    state={'schema':1,'algorithm':'masked-double-dqn','steps':D.STEPS,
           'contract_sha256':D.CONTRACT_SHA256,
           'online':{name:tensor.clone() for name,tensor in policy.network.state_dict().items()}}
    name=next(iter(state['online']))
    if change=='steps':state['steps']=D.STEPS+1
    elif change=='schema':state['schema']=True
    elif change=='extra':state['optimizer']={}
    elif change=='dtype':state['online'][name]=state['online'][name].double()
    elif change=='shape':state['online'][name]=state['online'][name].flatten()
    elif change=='missing':del state['online'][name]
    else:state['online'][name].flatten()[0]=float('nan')
    with pytest.raises(ValueError):D.validate_network_state(state,policy.network)


def test_recording_matches_evaluation_and_preserves_rng(policy,tmp_path):
    from ddqn_experiment.evaluation import GreedyPolicy
    base=generate_base(10200,'closed_ring');env=SymmetryWrapper(SiegeEnv(),transform=5)
    expected=play_episode(env,GreedyPolicy(policy.network),seed=0,options={'spec':base});env.close()
    random.seed(913);np.random.seed(217);torch.manual_seed(329)
    states=(random.getstate(),np.random.get_state(),torch.random.get_rng_state())
    actual=record_policy(policy,base_seed=base.seed,motif=base.motif,spec=base,view=5,deterministic=True)
    assert actual['actions']==expected['actions'] and actual['summary']==expected['summary']
    assert actual['visual_frames'] and b'streamlit:render' in html_bytes(actual)
    assert states[0]==random.getstate() and np.array_equal(states[1][1],np.random.get_state()[1])
    assert torch.equal(states[2],torch.random.get_rng_state())
    save_replay(actual,tmp_path/'replay.json')
    assert load_replay(tmp_path/'replay.json')['actions']==expected['actions']
    with pytest.raises(ValueError,match='greedy'):policy.predict({},action_masks=[],deterministic=False)


def test_custom_base_and_reserved_seed_guard(policy):
    from ui.custom_bases import custom_spec,layout_from_spec
    base=custom_spec(layout_from_spec(generate_base(42,'layered_core')))
    replay=record_policy(policy,base_seed=base.seed,motif='custom',spec=base,view=7,deterministic=True)
    assert replay['summary']['motif']=='custom' and replay['visual_frames']
    with pytest.raises(PermissionError):record_policy(policy,base_seed=11200,motif='layered_core',deterministic=True)


@pytest.fixture(scope='module')
def selected_policy():
    path,_=D.selected_checkpoint(ROOT)
    assert path is not None
    return D.load_selected_policy(path)


def test_exact_runtime_reference_and_loaded_tensors_keep_original_identity(selected_policy):
    assert hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest()==REFERENCE_SHA256
    assert REFERENCE['runtime']==recorded_runtime()
    assert REFERENCE['source_commit']=='ff36390a7f9756b3153394b1ea20f0bbf049afbb'
    assert REFERENCE['source_tree']=='0b7599631e184d4b2e023ff8e023c936955b257f'
    assert REFERENCE['identity']['policy_sha256']==D.POLICY_SHA256
    assert REFERENCE['identity']['training_contract_sha256']==D.CONTRACT_SHA256
    assert len(REFERENCE['cases'])==40
    path,_=D.selected_checkpoint(ROOT)
    contract=json.loads((path.parent.parent/'contract.json').read_text())
    assert REFERENCE['source_hashes']==contract['sources']
    original=torch.load(path,map_location='cpu',weights_only=True)['online']
    loaded=selected_policy.network.state_dict()
    assert set(loaded)==set(original)
    assert all(torch.equal(loaded[name],original[name]) for name in original)


@pytest.mark.parametrize('case',REFERENCE['cases'],ids=lambda case:f"{case['motif']}-view-{case['view']}")
def test_actual_inference_matches_original_strict_runtime_reference(selected_policy,case):
    spec=BaseSpec.from_json(json.dumps(case['spec']))
    replay=record_policy(selected_policy,base_seed=case['base_seed'],motif=case['motif'],
                         view=case['view'],spec=spec,deterministic=True)
    assert replay['actions']==case['actions']
    assert replay['summary']==case['summary']


@pytest.mark.parametrize('view',range(8))
def test_actual_selected_weights_replay_all_views(selected_policy,view):
    from ddqn_experiment.evaluation import GreedyPolicy
    base=generate_base(42,'closed_ring')
    env=SymmetryWrapper(SiegeEnv(),transform=view)
    try:
        expected=play_episode(env,GreedyPolicy(selected_policy.network),seed=0,options={'spec':base})
    finally:
        env.close()
    actual=record_policy(selected_policy,base_seed=base.seed,motif=base.motif,spec=base,
                         view=view,deterministic=True)
    assert actual['actions']==expected['actions']
    assert actual['summary']==expected['summary']


def test_actual_loader_cache_and_concurrent_sessions_preserve_rng(tmp_path):
    path,_=D.selected_checkpoint(ROOT)
    D._load_cached.clear()
    try:
        random.seed(913);np.random.seed(217);torch.manual_seed(329)
        states=(random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone())
        loaded=D.load_selected_policy(path)
        assert D.load_selected_policy(path) is loaded
        assert states[0]==random.getstate()
        assert np.array_equal(states[1][1],np.random.get_state()[1])
        assert torch.equal(states[2],torch.random.get_rng_state())
        copy=copy_selected_checkpoint(tmp_path/'selected')
        D.load_selected_policy(copy)
        assert D.load_selected_policy(path) is not loaded

        def replay(view):
            result=record_policy(loaded,base_seed=42,motif='closed_ring',view=view,deterministic=True)
            return result['actions'],result['summary']
        views=[0,3,0]
        expected=[replay(view) for view in views]
        with ThreadPoolExecutor(max_workers=2) as executor:
            actual=list(executor.map(replay,views))
        assert actual==expected and actual[0]==actual[2]
        assert states[0]==random.getstate()
        assert np.array_equal(states[1][1],np.random.get_state()[1])
        assert torch.equal(states[2],torch.random.get_rng_state())
    finally:
        D._load_cached.clear()


def test_actual_ui_generated_custom_base_views_and_downloads(selected_policy):
    from ui.custom_bases import layout_from_spec
    app=AppTest.from_file(str(ROOT/'ui/app.py')).run(timeout=30)
    def select(label,value):
        next(w for w in app.selectbox if w.label==label).set_value(value).run(timeout=30)
        assert not app.exception
    def generate():
        next(b for b in app.button if b.label=='Generate replay').click().run(timeout=30)
        assert not app.exception and not app.error
        return app.session_state['replay']
    select('Mode','Policy replay');select('Policy','Saved Masked DDQN')
    replay=generate()
    assert replay['policy']['model_sha256']==D.POLICY_SHA256
    assert replay['policy']['learner_seed']==100
    assert replay['policy']['training_contract_sha256']==D.CONTRACT_SHA256
    assert replay['acting']['mode']=='greedy-legal'
    assert len(app.get('download_button'))==2
    select('Agent view',5)
    assert 'replay' not in app.session_state
    assert generate()['acting']['view']==5
    app.session_state['custom_layout']=layout_from_spec(generate_base(42,'layered_core'),'DDQN real custom')
    select('Play on','My custom base')
    replay=generate()
    assert replay['summary']['motif']=='custom'
    assert replay['policy']['custom_base']['name']=='DDQN real custom'
    assert replay['visual_frames'] and b'streamlit:render' in html_bytes(replay)


def test_ui_selection_downloads_and_stale_replay(policy,monkeypatch):
    path=ROOT/'runs/ddqn/test-fixture/policies'/f'step-{D.STEPS:012d}.pt'
    details={'model_sha256':'TEST FIXTURE','actual_steps':D.STEPS,'training_contract_sha256':'TEST FIXTURE',
             'settings':{'condition':'DDQN','seed':100,'mode':'pilot'}}
    monkeypatch.setattr(D,'selected_checkpoint',lambda root:(path,details))
    monkeypatch.setattr(D,'load_selected_policy',lambda path:policy)
    app=AppTest.from_file(str(ROOT/'ui/app.py')).run(timeout=30)
    def select(label,value):
        next(w for w in app.selectbox if w.label==label).set_value(value).run(timeout=30)
        assert not app.exception
    select('Mode','Policy replay');select('Policy','Saved Masked DDQN')
    assert not any(w.label=='Action seed' for w in app.number_input)
    next(b for b in app.button if b.label=='Generate replay').click().run(timeout=30)
    assert not app.exception
    replay=app.session_state['replay']
    assert replay['policy']['model_version']=='DDQN' and replay['policy']['algorithm']=='masked-double-dqn'
    assert replay['policy']['training_transitions']==D.STEPS and replay['acting']['mode']=='greedy-legal'
    assert replay['visual_frames'] and len(app.get('download_button'))==2
    select('Agent view',1);assert 'replay' not in app.session_state
    select('Policy','Heuristic')
    next(b for b in app.button if b.label=='Generate replay').click().run(timeout=30)
    assert not app.exception and app.session_state['replay']['policy']['name']=='Heuristic'


def test_missing_checkpoint_disables_generation(monkeypatch):
    monkeypatch.setattr(D,'selected_checkpoint',lambda root:(None,{}))
    app=AppTest.from_file(str(ROOT/'ui/app.py')).run(timeout=30)
    next(w for w in app.selectbox if w.label=='Mode').set_value('Policy replay').run(timeout=30)
    next(w for w in app.selectbox if w.label=='Policy').set_value('Saved Masked DDQN').run(timeout=30)
    assert not app.exception
    assert next(b for b in app.button if b.label=='Generate replay').disabled


def test_public_app_offers_selected_ddqn():
    app=AppTest.from_file(str(ROOT/'ui/streamlit_app.py')).run(timeout=30)
    next(w for w in app.selectbox if w.label=='Mode').set_value('Policy replay').run(timeout=30)
    next(w for w in app.selectbox if w.label=='Policy').set_value('Saved Masked DDQN').run(timeout=30)
    assert not app.exception
    assert not next(b for b in app.button if b.label=='Generate replay').disabled
