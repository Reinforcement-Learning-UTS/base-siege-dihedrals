# Base Siege: foundation and masked TRPO

This is the first of three submission migration stages. It contains the shared
simulator, random and heuristic commanders, Gymnasium interface, reusable training
foundation, masked TRPO implementation, development monitor, fixed training and
development pools, dependency files, focused tests and canonical TRPO evidence.

Chandan leads review and migration of this stage, including his TRPO contribution.
The shared foundation retains its existing source provenance. This package does
not assign sole authorship of shared code.

## Set up and check

Use Python 3.12 and uv. Open a terminal in this folder, then run:

```bash
uv sync --frozen --python 3.12
uv run --frozen python -m pytest tests trpo/tests -q
uv run --frozen python -m trpo.runner --help
```

The tests include short synthetic CPU learning checks. They verify implementation
behaviour; they do not reproduce the completed scientific studies. The CLI help
checks that the training entrypoint imports without starting a training run.

## Contents and evidence

- `env/`, `agents/`, `train/` and `experiments/`: shared scientific foundation.
- `trpo/`: mask-aware TRPO, training entrypoint, checkpoint retention and tests.
- `version2/evaluation.py`: the development monitor required by TRPO training.
- `bases/train-v2.json` and `bases/dev-v2.json`: preserved fixed pool inputs.
- `provenance/`: pinned original-to-migrated source compatibility records.
- `tests/`: focused simulator, RL-contract and experiment-safeguard checks.
- `runs/study-v3/` and `runs/trpo-pilot-100/`: final and placement model references,
  training episode/TensorBoard log references, development histories and metadata.
- `artifacts/version3-final-evaluation/`, `artifacts/version3-mode-probe/` and
  `artifacts/trpo/selected-T1/`: completed results, frozen identities and selected
  model references.
- `docs/migration/stage1-trpo-evidence.json`: original file hashes and sizes,
  included evidence and the full inventory deferred to later archive migration.

The migration preserves the recorded scientific inputs and model-related module
names. Source compatibility records explain reviewed source changes while keeping
real file hashes truthful.

The evidence contains 124 canonical files representing 227,250,749 original bytes.
Of these, 63 large-file paths contain their original Git LFS pointers. Their real
model/log bytes are not included in this ZIP. The other 61 evidence files contain
the actual metadata and completed results. Model and log loading require later
central LFS upload and hydration against the approved hash inventory.

The preserved original workspace retains another 1,256 intermediate/recovery and
journal files, plus three duplicate ZIP exports. Those are inventoried for later
archive migration rather than included here. No original evidence was deleted.
Inference models are not a replacement for complete training recovery state.

The remaining PPO/DDQN workflows, full V2/V3 execution packages, model catalogs
and application arrive in later migration stages. This first stage has no web
application. Do not push the pointer commit until the team has verified that the
required LFS objects are available at the destination repository.

Read the separate instructions supplied beside the ZIP before repository upload.
