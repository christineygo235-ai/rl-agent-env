# Experiment report: CliffWalking-v1

## Method
SARSA (on-policy TD control, Sutton & Barto, Example 6.6). With exploration on,
SARSA learns the safe path away from the cliff; the greedy policy after
training follows it (return -17), whereas Q-learning would learn the shorter
cliff-edge path (-13).

## Hyperparameters
| Name | Value |
|---|---|
| episodes | 1000 |
| gamma | 1.0 |
| alpha | 0.5 |
| epsilon | linear 1.0 → 0 over the first 80 % of episodes |
| max steps per episode | 200 |
| seed | {train_seed} |

## Experiments
{runs_table}

## Results
Mean return of the greedy policy on {dev_episodes} dev episodes, evaluated by the
harness: **{dev_mean:.1f}** (threshold {threshold}).

## Limitations
- The environment is deterministic (not slippery), so the greedy return is exact;
  the slippery variant would need more exploration and a different threshold.
