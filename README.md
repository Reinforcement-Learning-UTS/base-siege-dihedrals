# Base Siege

Base Siege studies army selection and troop deployment in a custom reinforcement
learning environment. It contains Maskable PPO, masked Double DQN and Maskable
TRPO, reproducible experiment workflows, trained-model references, notebooks and
preserved research outputs. The Streamlit application is the next migration stage.

Developed for 43008 Reinforcement Learning at the University of Technology Sydney
by Chandan, Sreekar and Vaibhav. Original code contributions and experiment evidence
retain their provenance from [Vaibhav's repository](https://github.com/vaibhavbai1/base-siege),
commit `68755c84`. Migration responsibility does not imply sole authorship of shared
source or new execution of historical experiments.

## Setup and quick checks

Use Python 3.12.14 for the recorded research checks, [uv](https://docs.astral.sh/uv/)
and Git LFS. Keep the full research
archive out of an ordinary source checkout until its files are needed:

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Reinforcement-Learning-UTS/base-siege-dihedrals.git
cd base-siege-dihedrals
uv sync --frozen --python 3.12.14
uv run --frozen python -m pytest tests trpo/tests ddqn_experiment/tests -q
uv run --frozen python -m ddqn_experiment --help
```

Run commands beside `pyproject.toml` at the repository root. The tests include
short synthetic CPU learner and recovery checks. CLI help imports the training
entrypoint without starting a training cohort or final scientific evaluation.

## Repository structure

| Path | Purpose |
|---|---|
| `env/`, `agents/` | Simulator, action masking, symmetry and baseline commanders |
| `train/`, `experiments/` | Shared PPO, rollout, checkpoint and recovery infrastructure |
| `trpo/`, `ddqn_experiment/` | Masked TRPO and Double DQN implementations |
| `version2/`, `v2_*/`, `v3_*/`, `ddqn_final_benchmark/` | Versioned studies and evaluation workflows |
| `bases/` | Preserved layout pools and registrations |
| `runs/`, `artifacts/`, `version2-records/` | Model references, raw histories, frozen protocols and completed results |
| `ui/` | Model catalogs, CPU inference adapters and recorded-replay support; application entrypoints arrive next |
| `notebooks/`, `tools/`, `final_evaluation_bundle/` | Research notebooks, inspection, checks and replay/report utilities |
| `docs/`, `provenance/` | Technical guides, handbooks, evidence inventories and pinned source compatibility |
| `reference/` | Preserved earlier source, console logs, handoffs and application snapshots |

## Models and seeds

The final collection contains **51 unique checkpoints**, totalling **212,250,251
bytes**: 20 V1 PPO models, 20 V2 PPO models, ten TRPO learners and the selected
DDQN seed-100 checkpoint. The [model guide](docs/MODELS.md) explains their identities,
runtime requirements and limits. All reviewed LFS objects are available in the
organisation store; pointer files are not model weights.

After committing the source, fetch the final-model allowlist only when model
verification is needed:

```bash
git lfs pull --include="$(cat docs/migration/final-models-lfs-include.txt)"
mkdir -p local-output
uv run --frozen python -m tools.check_demo_models --out local-output/final-model-check.json
```

The include list covers aliases of the same approved final weights, so catalog
selection cannot pick an unhydrated duplicate. It does not fetch the full research
archive. Each model check loads actual weights and records a generated non-final
example; it is not a new scientific benchmark.

The complete research test suite also loads supplied checkpoints. After the
selective fetch, run it with the recorded Python 3.12.14 runtime:

```bash
uv run --frozen python -m pytest -q tests trpo ddqn_experiment ddqn_final_benchmark \
  version2 v2_cohort_evaluation v2_development v2_final v2_phases v2_replays \
  v2_replication v3_final v3_probe
```

## Experiments and results

Completed evidence includes the V1 and V2 PPO studies, the selected DDQN benchmark,
the frozen V3 PPO/TRPO evaluation and a separate exploratory acting-mode probe.
See [results and evidence](docs/RESULTS.md) for canonical paths and report-figure
inputs. Raw records, frozen protocols and completion hashes take precedence over
informal summaries.

The studies differ in learner counts, budgets and acting modes. DDQN has one
selected learner; the exploratory TRPO probe is separate from frozen V3 DS
evaluation. These results do not establish a controlled three-algorithm ranking.

All supplied checkpoints, histories and outputs retain their original bytes.
Recovery generations and duplicate exports are intentionally retained as research
evidence. The [research inventory](docs/migration/stage2-research-inventory.json)
records the package's exact base commit, file hashes and original LFS identities.

## Application and release status

This stage supplies the research workflows and model/replay adapters. The complete
Streamlit entrypoints, interactive editor, application tests, CI integration and
hosted configuration follow in the application stage. Do not use an old archived
application snapshot as the current entrypoint.

Local full-project preparation has already exercised PPO, DDQN and TRPO with all
51 final models. Deployment still requires the final application migration and
verification of the actual hosted CPU runtime and selected-model hydration.

## Documentation

- [Documentation and original handbooks](docs/README.md)
- [Models and learner seeds](docs/MODELS.md)
- [Results and report figures](docs/RESULTS.md)
- [Research runbook](docs/RUNBOOK.md)
- [Migration and verification status](docs/SUBMISSION_STATUS.md)
