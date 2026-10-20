# Research in SageMaker Studio

Use persistent project storage for long-running experiments. The research
notebooks call shared Python modules; they do not replace the command-line
contracts, registrations or completed evidence. The Streamlit hosting profile is
described separately in [deployment](DEPLOYMENT.md).

## Prepare the checkout

Install Git LFS and uv using their official instructions, then run the bootstrap
from a terminal. It clones the organization repository with automatic LFS download
disabled, fetches the approved final checkpoints and builds the locked environment.
For an existing checkout, pass `--skip-clone --dir /absolute/project/path`.

```bash
bash sagemaker_bootstrap.sh
```

The default location is `$HOME/base-siege`. `--lfs all` fetches the complete archive;
its unique LFS content is approximately 9.89 GiB and requires additional space for
working copies and dependencies. `--lfs none` prepares source without model checks.
A new run also needs storage for recovery generations and growing episode logs.
Check available capacity before choosing a training budget.

## Runtime and compute

The recorded scientific checks use Python 3.12.14. Frozen contracts also bind
package versions, source and inputs. A different GPU/runtime is a new execution
context; do not edit a historical contract to make it pass. Application-only CPU
adapters have a documented, narrower compatibility profile in [models](MODELS.md).

The simulator runs on CPU even when neural-network operations use CUDA. Measure
short CPU and GPU runs in the chosen environment before starting a cohort. This
repository does not create AWS resources or select a paid instance automatically.

## Notebooks and recovery

Select the `Python 3 (Base Siege)` kernel installed by the bootstrap and start with
`notebooks/00_sagemaker_setup.ipynb`. Training switches are off by default. Use the
[runbook](RUNBOOK.md) for pool, pilot, cohort and evaluation commands.

Keep the complete run directory, including input pools, contract, manifests,
checkpoint generations and committed history. PPO recovery resumes at completed
optimizer updates; an interrupted rollout is collected again. Checksummed atomic
generations allow recovery to fall back from a damaged newest generation. TRPO
retains its own snapshots and journals. Complete DDQN training recovery is absent.
A single-writer lock prevents concurrent writers to one run.

Exact local recovery tests do not establish cross-hardware bitwise equality.
Keep original evidence untouched and use a fresh destination for new checks.

## Interactive app in Studio

```bash
bash sagemaker_app.sh --dir /absolute/project/path
```

The launcher prints the JupyterLab proxy path and preserves Streamlit's CORS and
XSRF protections. Proxy routing can differ by Studio configuration; switch between
`--mode absolute` and `--mode plain` if necessary, then check the component and
websocket requests. A local or Community Cloud app does not need this proxy
launcher. No hosted Studio verification is claimed by the local source checks.
