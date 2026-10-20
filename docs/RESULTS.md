# Results and evidence

Completed results are stored as CSV/JSON records with protocols, completion receipts and output hashes. The submission cleanup does not repeat training or final scientific evaluation.

| Study | Canonical evidence |
|---|---|
| V1 sampled PPO | `artifacts/sampled-final-v1/`, `final_evaluation_bundle/` |
| V2 PPO final evaluation | `artifacts/version2-final-evaluation/`, `v2_final/` |
| Selected DDQN on V2 benchmark | `artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/` |
| Frozen V3 PPO/TRPO DS evaluation | `artifacts/version3-final-evaluation/`, `v3_final/` |
| Exploratory acting-mode probe | `artifacts/version3-mode-probe/`, `v3_probe/` |

PPO includes ten learners per condition. DDQN represents one selected learner with a different checkpoint budget and acting procedure. Keep the exploratory TRPO SS probe separate from frozen V3 DS evaluation in reports, plots and application labels.

## Report figures

Recorded episode histories, development diagnostics and final CSVs support training-return curves, development destruction curves, learner distributions, final condition comparisons and layout-family summaries. TensorBoard supports diagnostic plots where those values were actually recorded.

Label study versions and acting modes. Resumed TRPO runs contain repeated diagnostic steps; document an attempt/step policy before plotting. DDQN has periodic progress records but lacks complete raw training episode histories.

Historical numerical summaries remain in `reference/upstream/README.md` and the original Word handbooks. Raw records, frozen protocols and completion hashes take precedence over informal handoff prose.
