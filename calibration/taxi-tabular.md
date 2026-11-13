# Calibration: taxi-tabular v1

**ADMITTED**

| Criterion | Value |
|---|---|
| Seeds | 10 |
| Reference pass rate | 100% (required ≥ 90%) |
| Random policy fails on every seed | True |
| Deterministic (same seed twice) | True |
| Score unit | mean return |
| Threshold (task) | 7.5 |
| Gymnasium `reward_threshold` (canonical env) | 8 |
| Reference score min / median / max | 7.765 / 7.975 / 8.185 |
| Slowest reference re-run | 2.7 s |
| Suggested train timeout (3×, rounded) | 30 s |

| Seed | Pass | Reward | Score | Random | Re-run s |
|---|---|---|---|---|---|
| 1234 | True | 0.900 | 8.008 | -775.625 | 2.6 |
| 2234 | True | 0.900 | 7.984 | -767.721 | 2.7 |
| 3234 | True | 0.900 | 7.829 | -771.013 | 2.7 |
| 4234 | True | 0.900 | 7.989 | -772.594 | 2.6 |
| 5234 | True | 0.900 | 7.979 | -770.554 | 2.7 |
| 6234 | True | 0.900 | 7.898 | -771.731 | 2.7 |
| 7234 | True | 0.900 | 7.971 | -770.502 | 2.7 |
| 8234 | True | 0.900 | 8.185 | -769.059 | 2.7 |
| 9234 | True | 0.900 | 7.935 | -767.638 | 2.6 |
| 10234 | True | 0.900 | 7.765 | -771.311 | 2.6 |
