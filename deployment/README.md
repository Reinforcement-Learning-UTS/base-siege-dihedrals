# Base Siege interactive demo

Base Siege explores army selection and troop deployment in a custom reinforcement
learning simulator. This application supports manual play, a base editor, learned
policy replays and JSON/HTML replay downloads.

The curated collection contains 40 PPO models, ten TRPO learners and one selected
Double DQN checkpoint. All 51 weights and their identity records are included.
Models load on demand into bounded CPU caches. Generated and development layouts
are available; final evaluation layouts are not exposed through the app.

Use Python 3.12 from this directory:

```bash
python -m pip install -r ui/requirements.txt
python -m streamlit run ui/streamlit_app.py
```

To validate all models and the app before hosting:

```bash
python -m pip install -r deployment/requirements-test.txt
python -m pytest -q tests/test_public_app.py
python tools/check_public_demo.py --report public-check.json
```

For Streamlit Community Cloud, choose `ui/streamlit_app.py`, Python 3.12 and no
secrets. Preserve the model files and original sidecars. The app contains no
training or cloud-write controls. See [deployment](docs/DEPLOYMENT.md) and
[model identities](docs/MODELS.md) for details and limitations.

Developed by Chandan, Sreekar and Vaibhav for UTS 43008 Reinforcement Learning.
