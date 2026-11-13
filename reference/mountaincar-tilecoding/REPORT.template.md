# Experiment report: MountainCar-v0

## Method
Semi-gradient SARSA with tile coding (Sutton & Barto, section 10.1, Example
10.1). Linear action values over 8 grid tilings of position and velocity with
asymmetric offsets. Exploration comes from optimistic initial values (weights 0,
rewards -1), so epsilon = 0. Every 50 episodes the greedy policy is scored on
10 training-side seeds and the best weights are kept.

## Hyperparameters
| Name | Value |
|---|---|
| episodes | 1000 |
| tilings | 8 × (8×8) tiles, displacement (1, 3) |
| alpha | linear 0.5/8 → 0.1/8 |
| gamma | 1.0 |
| epsilon | 0 (optimistic initialisation) |
| checkpoint selection | best of every 50 episodes on 10 training-side seeds |
| seed | {train_seed} |

## Experiments
{runs_table}

## Results
Mean return of the greedy policy on {dev_episodes} dev episodes, evaluated by
the harness: **{dev_mean:.1f}** (threshold {threshold}).

## Limitations
- With a constant step size, more episodes made the policy worse (oscillating
  weights); the decaying step size and checkpoint selection fix that, but at a
  slightly higher compute cost.
- Grid tiling ignores structure that hashing (IHT) could exploit for larger
  state spaces; it is sufficient for 2-D MountainCar.
