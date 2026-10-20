# Documentation

Start with the [project README](../README.md), [models](MODELS.md),
[results](RESULTS.md) and [current migration status](SUBMISSION_STATUS.md).

## Research guides

- [Architecture](ARCHITECTURE.md)
- [Research runbook](RUNBOOK.md)
- [Experiment plan](EXPERIMENT_PLAN.md)
- [Scientific decisions](DECISIONS.md)
- [TRPO interface](TRPO_INTERFACE.md)
- [TRPO evidence import](TRPO_IMPORT.md)

The architecture and research runbook also describe the intended complete project.
Application entrypoints and hosted deployment arrive in the following stage.

## Original handbooks

The three supplied documents remain byte-identical historical references. Their
prose does not override frozen protocols, completion records or output hashes.

- [PPO V1](Base_Siege_Complete_Experiment_Handbook.docx)
- [PPO V1/V2](Base_Siege_Complete_Experiment_Handbook_V1_V2.docx)
- [DDQN/TRPO](Base_Siege_DDQN_TRPO_Complete_Experiment_Handbook.docx)

## Preservation records

- [Final model identities](migration/final-models.json)
- [First-stage TRPO inventory](migration/stage1-trpo-evidence.json)
- [Research migration inventory](migration/stage2-research-inventory.json)
- [Original-to-reference relocations](migration/relocations.json)

The first-stage inventory's publication field describes its pre-upload snapshot.
The central LFS upload and final-model readback have since passed. Earlier source,
console logs, application copies and handoffs remain under `reference/`; they are
historical evidence rather than active entrypoints.
