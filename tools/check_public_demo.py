"""Check the complete CPU demo, its selectors and all approved final models."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import sys


def check(root, *, all_models=True):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    import torch
    from streamlit.testing.v1 import AppTest
    from ui.v2_models import published_catalog
    from ui import ddqn_models, trpo_models

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12 for the recorded inference dependencies.')
    if sys.platform == 'linux' and torch.version.cuda is not None:
        raise RuntimeError('The Linux demo must use the CPU-only PyTorch wheel.')
    catalog, issues = published_catalog(root)
    if issues or len(catalog) != 40:
        raise RuntimeError('The forty final PPO models are not ready: ' + '; '.join(issues))
    trpo = trpo_models.application_catalog(root)
    if {item['settings']['seed'] for _, item in trpo} != set(range(10)):
        raise RuntimeError('All ten final TRPO learners must be available.')
    ddqn, _ = ddqn_models.selected_checkpoint(root)
    if ddqn is None:
        raise RuntimeError('The selected DDQN checkpoint is missing.')
    app = AppTest.from_file(str(root / 'ui/streamlit_app.py')).run(timeout=30)
    if app.exception or app.title[0].value != 'Base Siege':
        raise RuntimeError('The Streamlit app did not render.')
    def select(label, value):
        next(w for w in app.selectbox if w.label == label).set_value(value).run(timeout=30)
        if app.exception:
            raise RuntimeError('The app failed after changing ' + label)
    select('Mode', 'Policy replay')
    policy_widget = next(w for w in app.selectbox if w.label == 'Policy')
    expected = {'Heuristic', 'Valid random', 'Saved Maskable PPO',
                'Saved Masked DDQN', 'Saved Maskable TRPO'}
    if set(policy_widget.options) != expected:
        raise RuntimeError('The hosted profile does not offer every method.')
    select('Policy', 'Heuristic')
    next(b for b in app.button if b.label == 'Generate replay').click().run(timeout=30)
    if app.exception or not app.session_state['replay']['summary']['finish_reason']:
        raise RuntimeError('The baseline replay did not complete.')
    for policy in ('Saved Maskable PPO', 'Saved Masked DDQN', 'Saved Maskable TRPO'):
        select('Policy', policy)
        next(b for b in app.button if b.label == 'Generate replay').click().run(timeout=30)
        if app.exception or not app.session_state['replay']['summary']['finish_reason']:
            raise RuntimeError('The learned replay did not complete: ' + policy)
        if len(app.get('download_button')) != 2:
            raise RuntimeError('Both JSON and HTML exports must be available.')
    model_report = None
    if all_models:
        from tools.check_demo_models import check as check_models
        from tools.check_demo_models import ROOT as model_root
        if model_root != root:
            raise RuntimeError("Run the preflight in a fresh process from the bundle directory.")
        model_report = check_models()
        if model_report['status'] != 'passed' or model_report['passed_models'] != 51:
            raise RuntimeError('One or more final model examples failed.')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'status': 'passed', 'python': sys.version.split()[0], 'platform': sys.platform,
            'torch': importlib.metadata.version('torch'), 'cuda_runtime': torch.version.cuda,
            'peak_process_rss_mib': round(peak / (1024 ** 2 if sys.platform == 'darwin' else 1024), 1),
            'ppo_models': len(catalog), 'trpo_models': len(trpo), 'ddqn_models': 1,
            'learned_app_replays': 3, 'json_and_html_exports': 'passed',
            'all_final_model_examples': model_report,
            'scope': 'Local CPU software checks; not a capacity or scientific performance benchmark'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = check(args.root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'all_final_model_examples'}, indent=2))
