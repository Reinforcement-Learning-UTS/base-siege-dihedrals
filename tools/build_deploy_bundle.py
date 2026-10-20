"""Build a self-contained Streamlit source bundle with the approved final models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    '.gitattributes', '.lfsconfig', '.streamlit/config.toml', 'conftest.py',
    'ui/streamlit_app.py', 'ui/app.py', 'ui/deployment.py', 'ui/policy_playback.py',
    'ui/manual_game.py', 'ui/custom_bases.py', 'ui/v2_models.py', 'ui/v2_checkpoints.json',
    'ui/trpo_models.py', 'ui/v3_checkpoints.json', 'ui/ddqn_models.py',
    'ui/evaluation_tables.py', 'ui/requirements.txt',
    'ui/editor/index.html', 'ui/editor/editor.js', 'ui/editor/editor.css', 'ui/editor/glyphs.js',
    'ui/frontend/index.html', 'ui/frontend/app.js', 'ui/frontend/manual_engine.js',
    'ui/frontend/theme.css', 'ui/frontend/playback.css', 'ui/frontend/glyphs.js',
    'tools/export_replay_html.py', 'tools/check_public_demo.py', 'tools/check_demo_models.py',
    'provenance/__init__.py', 'provenance/compatibility.py', 'provenance/_record_pin.py',
    'provenance/source-compatibility.json', 'models/catalog.json', 'models/lfs-paths.txt',
    'bases/dev-v2.json', 'demo_assets/README.md', 'docs/DEPLOYMENT.md', 'docs/MODELS.md',
    'tests/test_public_app.py', 'deployment/requirements-test.txt',
)
# These exact source files are checked by the original inference contracts.
SOURCE_PACKAGES = ('env', 'agents', 'train', 'experiments', 'ddqn_experiment', 'trpo')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe bundle path: {name}')
    path = root / relative
    if (not path.is_file() or not path.resolve().is_relative_to(root) or
            any((root / part).is_symlink() for part in (relative, *relative.parents))):
        raise ValueError(f'Missing file or symbolic link in bundle input: {name}')
    return path


def build_bundle(output, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError('Choose a separate bundle directory.')
    if output.exists() or output.with_suffix('.zip').exists():
        raise FileExistsError('Choose a new bundle folder; previous bundles are preserved.')
    files = {name: name for name in SOURCE_FILES}
    for folder in SOURCE_PACKAGES:
        for path in sorted((root / folder).glob('*.py')):
            name = path.relative_to(root).as_posix()
            files[name] = name
    files['deployment/streamlit-ci.yml'] = '.github/workflows/ci.yml'
    files['deployment/README.md'] = 'README.md'
    inventory = json.loads(checked_file(root, 'models/catalog.json').read_text())
    models = inventory['models']
    counts = {algorithm: sum(item['algorithm'] == algorithm for item in models)
              for algorithm in ('MaskablePPO', 'MaskableTRPO', 'masked-double-dqn')}
    if counts != {'MaskablePPO': 40, 'MaskableTRPO': 10, 'masked-double-dqn': 1}:
        raise ValueError('The deployment needs all 51 approved final models.')
    if len({item['sha256'] for item in models}) != 51:
        raise ValueError('Final checkpoint identities are duplicated.')
    for item in models:
        path = checked_file(root, item['path'])
        if path.stat().st_size != item['size'] or digest(path) != item['sha256']:
            raise ValueError(f"Checkpoint is missing, changed or still an LFS pointer: {item['path']}")
        files[item['path']] = item['path']
        if item.get('manifest_path'):
            name = item['manifest_path']
            if digest(checked_file(root, name)) != item['manifest_sha256']:
                raise ValueError(f'Original manifest checksum differs: {name}')
            files[name] = name
        if item['algorithm'] == 'MaskableTRPO':
            name = str(Path(item['path']).with_name('contract.json'))
            files[name] = name
        elif item['algorithm'] == 'masked-double-dqn':
            name = item['contract_path']
            files[name] = name
            name = str(Path(item['path']).with_suffix('.json'))
            files[name] = name
    for path in sorted((root / 'demo_assets/replays').glob('*.json')):
        name = path.relative_to(root).as_posix()
        files[name] = name
    # Check every source before creating output, so a failed build leaves no partial bundle.
    sources = {name: checked_file(root, name) for name in files}
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source, target in files.items():
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[source], destination)
        hashes[target] = digest(destination)
    manifest = {'schema_version': 2, 'final_models': 51,
                'unique_checkpoint_bytes': sum(item['size'] for item in models),
                'algorithms': counts, 'files': hashes}
    (output / 'bundle-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    return {'folder': str(output), 'zip': str(output.with_suffix('.zip')),
            'files': len(hashes), 'final_models': 51,
            'unique_checkpoint_bytes': manifest['unique_checkpoint_bytes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/streamlit-deploy')
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.out), indent=2))


if __name__ == '__main__':
    main()
