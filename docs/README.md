# Documentation

Start with the [project overview](../README.md), then choose the guide that fits
the task.

| Guide | Purpose |
|---|---|
| [Architecture](ARCHITECTURE.md) | Simulator, decisions, policies, recovery and evaluation |
| [Models](MODELS.md) | All 51 checkpoints, learner seeds and runtime requirements |
| [Results](RESULTS.md) | Canonical evidence, interpretation and report figures |
| [Runbook](RUNBOOK.md) | Research commands and safeguards |
| [Deployment](DEPLOYMENT.md) | CPU app, standalone bundle and Streamlit settings |
| [SageMaker Studio](SAGEMAKER_STUDIO.md) | Long-running training and recovery workflow |

## Experiment handbooks

These supplied documents are preserved historical references. Frozen protocols,
completion records and output hashes take precedence over descriptive summaries.

- [PPO V1](Base_Siege_Complete_Experiment_Handbook.docx)
- [PPO V1 and V2](Base_Siege_Complete_Experiment_Handbook_V1_V2.docx)
- [DDQN and TRPO](Base_Siege_DDQN_TRPO_Complete_Experiment_Handbook.docx)

The machine-readable [model catalog](../models/catalog.json) records final weight
paths, hashes, sizes and learner identities. The pinned records in `provenance/`
recognise exact reviewed source identities without changing scientific evidence.
