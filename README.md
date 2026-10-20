# Base Siege

Base Siege studies army selection and troop deployment in a custom reinforcement
learning environment. This repository currently contains the simulator, baseline
commanders, shared training infrastructure, masked TRPO implementation and
preserved TRPO experiment evidence.

Developed for 43008 Reinforcement Learning at the University of Technology Sydney
by Chandan, Sreekar and Vaibhav. Chandan contributed the TRPO implementation. Shared
source and recorded experiments retain their original provenance from
[Vaibhav's repository](https://github.com/vaibhavbai1/base-siege), commit `68755c84`.

## Setup

Use Python 3.12, [uv](https://docs.astral.sh/uv/) and Git LFS. Clone without
automatically downloading research models and logs:

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Reinforcement-Learning-UTS/base-siege-dihedrals.git
cd base-siege-dihedrals
uv sync --frozen --python 3.12
```

Run these commands from the repository root, beside `pyproject.toml`:

```bash
uv run --frozen python -m pytest tests trpo/tests -q
uv run --frozen python -m trpo.runner --help
```

The checks run short CPU examples and import the training entrypoint. They do not
start a full experiment or require the supplied model binaries.

## Repository structure

| Path | Contents |
|---|---|
| `env/` | Simulator, legal actions, combat, rendering and symmetry |
| `agents/` | Random and heuristic baseline commanders |
| `train/`, `experiments/` | Shared policy, rollout, experiment and recovery infrastructure |
| `trpo/` | Masked TRPO, training entrypoint, retention and tests |
| `version2/evaluation.py` | Development monitor used by TRPO training |
| `bases/` | Preserved training and development pools |
| `runs/`, `artifacts/` | Canonical TRPO model references, histories and completed results |
| `provenance/` | Pinned original-to-migrated source compatibility records |
| `tests/` | Simulator, RL contract and experiment checks |
| `docs/migration/` | Evidence inventory and preservation records |

## Models and experiment evidence

The current evidence inventory contains 124 canonical TRPO files, representing
227,250,749 original bytes. Sixty-three paths use Git LFS pointers for 62 distinct
objects; the other files contain the actual metadata and completed results.
All required objects are uploaded and verified in this repository's LFS store.

Download the ten final TRPO model files only when needed:

```bash
git lfs pull --include="runs/study-v3/*/model.zip"
```

The corresponding run directories retain placement models, episode and
TensorBoard histories, development records and manifests. Completed evaluations
are under `artifacts/version3-final-evaluation/` and
`artifacts/version3-mode-probe/`. Selected-model evidence is under
`artifacts/trpo/selected-T1/`.

The [TRPO evidence inventory](docs/migration/stage1-trpo-evidence.json) records
original paths, sizes and hashes. Its publication-status field describes package
preparation before the central LFS upload. The objects are now available.
Another 1,256 recovery/intermediate records and three duplicate exports remain
preserved in the source workspace for later research-archive migration.

Scientific inputs, model identities and output bytes are preserved. Reviewed
source cleanup is recorded explicitly in `provenance/`; training checkpoints and
historical results are not rewritten to accommodate cleanup.

## Migration status

The foundation and TRPO stage is present. Remaining PPO/DDQN workflows, V2/V3
research packages, catalogs, results and report tooling arrive in the next stage.
The complete Streamlit application and hosted release follow separately.

This stage has no web application. The two original contribution commits remain
in the repository history; the layout correction places the project directly at
the root so setup commands, imports and later packages use the same paths.
