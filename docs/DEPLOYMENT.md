# Streamlit deployment

The deployment entrypoint is `ui/streamlit_app.py`. It uses the shared app and
simulator, with all 51 final models available through a curated catalog. Inference
runs on CPU. No training job, application secret or paid API is required.

## Verify locally

Use Python 3.12 and install `ui/requirements.txt`. Linux uses a hash-pinned CPU
PyTorch wheel. Keep the root research lockfile for research environments; it is
not part of the standalone app bundle.

```bash
python -m pip install -r ui/requirements.txt -r deployment/requirements-test.txt
python -m streamlit run ui/streamlit_app.py
python tools/check_public_demo.py --report local-output/public-check.json
python tools/build_deploy_bundle.py --out local-output/streamlit-app
```

Create `local-output/` first. The builder refuses to replace an existing folder
or ZIP. It verifies model hashes before copying anything. The bundle includes
all 40 PPO, ten TRPO and one DDQN checkpoints, original sidecars, the development
pool, UI, simulator and the source files required by frozen inference contracts.
Small training modules remain for source identity checks; the app does not execute
training. Raw histories, intermediate recovery generations, notebooks and final
layout pools are excluded from the app bundle and retained in the research repo.

Run the bundle from its own directory with a fresh environment, without adding
the research checkout to `PYTHONPATH`:

```bash
cd local-output/streamlit-app
python -m pip install -r ui/requirements.txt -r deployment/requirements-test.txt
python -m pytest -q tests/test_public_app.py
python tools/check_public_demo.py --report public-check.json
```

The preflight loads every final model and completes generated examples. It also
checks manual play, baseline and learned replay controls, and replay downloads.
These checks are local evidence, not measurements of the hosted service.

## Community Cloud settings

| Setting | Value |
|---|---|
| Entry file | `ui/streamlit_app.py` |
| Python | 3.12 |
| Requirements | `ui/requirements.txt` |
| Secrets | None |
| Hardware | CPU |

Deploy the standalone bundle's contents from a dedicated repository so the host
does not download the research archive. Model weights total 212.25 MB before
sidecars and dependencies. Preserve `.gitattributes` and use Git LFS for the
weights in that repository. Confirm that its LFS objects are available before
starting the build. Choose access settings deliberately when connecting the
repository; this source preparation does not change repository or app visibility.

Each session owns its game and action RNG. Process caches hold at most two PPO,
two TRPO and one DDQN policy, with serialised predictions and one CPU thread.
Visitors can select every approved learner without keeping all learners in RAM.
Filesystem writes and sessions are temporary on the host. Download important
replays before leaving.

After a real deployment, check installation logs, all three learned methods,
manual shopping/deployment, the editor, development layouts, saved replays,
JSON/HTML downloads and restart behaviour. Measure memory and response time with
a few simultaneous visitors. Keep the local app available for presentations;
free hosting can sleep and its resource limits can change.

## Continuous integration

`.github/workflows/ci.yml` uses read-only repository permissions. The macOS ARM
job checks the exact recorded research runtime. The Linux job checks the CPU app,
all final models and an independent standalone bundle. The bundle has its own
Linux workflow. Neither workflow publishes an app or writes to another repo.
Community Cloud updates from its connected branch; use passing required checks
and reviewed changes to control releases.

## Provider documentation

- [Dependency selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [File organisation and Git LFS](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Managing app resources](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
