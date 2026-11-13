# Experiment report: Taxi-v4

## Method
Tabular Q-learning (Sutton & Barto, section 6.5) with an epsilon-greedy
behaviour policy; the final policy is greedy w.r.t. the Q-table (`q.npy`, 500x6).

## Hyperparameters
| Name | Value |
|---|---|
| episodes | 5000 |
| gamma | 0.99 |
| alpha | 0.2 (constant) |
| epsilon | linear 1.0 → 0 over the first 80 % of episodes |
| seed | {train_seed} |

## Experiments
{runs_table}

## Results
Mean return of the greedy policy on {dev_episodes} dev episodes, evaluated by the
harness: **{dev_mean:.3f}** (threshold {threshold}).

## Limitations
- The optimal mean return depends on the start-state distribution (about 8–8.5);
  differences of ±0.1 over a few hundred episodes are noise.
