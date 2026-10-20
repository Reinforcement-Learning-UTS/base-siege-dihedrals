# Results and evidence

The repository contains completed CSV/JSON results, frozen protocols, completion
receipts and output hashes. App replays are demonstrations on non-final layouts;
they are not additional benchmark observations.

| Study | Canonical evidence |
|---|---|
| V1 sampled PPO | `artifacts/sampled-final-v1/`, `final_evaluation_bundle/` |
| V2 PPO final evaluation | `artifacts/version2-final-evaluation/`, `v2_final/` |
| Selected DDQN on V2 benchmark | `artifacts/ddqn/selected-lr3e5-seed100-v2-benchmark/` |
| Frozen V3 PPO/TRPO DS evaluation | `artifacts/version3-final-evaluation/`, `v3_final/` |
| Exploratory acting-mode probe | `artifacts/version3-mode-probe/`, `v3_probe/` |

## Reading the findings

V2 PPO used ten learners in each condition over 235 final layouts. P achieved
92.96% mean destruction and 74.26% full clears; PA achieved 93.53% and 76.72%.
The corresponding heuristic achieved 56.90% and 2.29%.

V3 used 241 final layouts. Its frozen DS evaluation records 92.95% destruction
for P, 93.63% for PA and 56.11% for T. Full-clear rates are 74.25%, 77.07% and
7.54%. The heuristic achieved 55.09% destruction and 3.53% full clears.
Both summaries come from each study's `condition_summary.csv`, `all_final` rows.

The separate V3 mode probe recorded TRPO destruction of 56.11% under DS and
90.85% under SS, with all ten learners improving. This exploratory observation
explains the app's SS default for TRPO. It is not a replacement for the frozen DS
benchmark. PPO's mode-probe mean changed from 92.95% to 92.40%.

The selected DDQN learner, seed 100 at 2,097,152 transitions, recorded 54.41%
destruction and 21.70% full clears over 1,880 episodes on the V2 benchmark.
Its budget and greedy acting procedure differ from PPO, and it has one selected
learner. There is no across-learner uncertainty estimate for this checkpoint.
The evidence does not support a controlled three-algorithm ranking.

## Figures for the report

Recorded histories, diagnostics and final tables support training-return curves,
development destruction curves, learner distributions and layout-family summaries.
Use the canonical episode and learner tables to calculate the plotted quantities;
keep learner seeds as the unit of uncertainty rather than treating games as
independent learners. Label the study, acting mode, metric and pool on each figure.

TRPO resumed runs contain repeated diagnostic steps. Define a consistent
attempt/step policy before plotting. DDQN has periodic progress records but lacks
complete raw training episode histories. TensorBoard can support diagnostic plots
only for values that were actually logged. None of these limitations should be
filled with invented training data.

## Interpretation limits

The simulator uses simplified combat and partial observations. Its target and
pathfinding rules can leave troops unable to reach useful targets. Observed results
are conditional on the frozen pools, budgets and acting procedures. Evaluation
contracts, raw records and completion hashes take precedence over the historical
handbooks if their prose differs.
