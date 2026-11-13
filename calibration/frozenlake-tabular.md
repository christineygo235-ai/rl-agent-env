# Calibration: frozenlake-tabular v3

**ADMITTED**

| Criterion | Value |
|---|---|
| Seeds | 10 |
| Reference pass rate | 100% (required ≥ 90%) |
| Random policy fails on every seed | True |
| Deterministic (same seed twice) | True |
| Score unit | worst hidden instance, normalised (random 0, threshold 1) |
| Threshold (task) | 0.7 (1.0 = fraction × optimal per instance) |
| Gymnasium `reward_threshold` (canonical env) | 0.7 |
| Reference score min / median / max | 1.091 / 1.112 / 1.132 |
| Reference / optimal per hidden instance min / median / max | 0.977 / 1.000 / 1.045 |
| Slowest reference re-run | 13.1 s |
| Suggested train timeout (3×, rounded) | 40 s |

| Seed | Pass | Reward | Score | Random | Re-run s |
|---|---|---|---|---|---|
| 1234 | True | 0.900 | 1.132 | 0.000 | 7.0 |
| 2234 | True | 0.900 | 1.091 | 0.000 | 8.4 |
| 3234 | True | 0.900 | 1.093 | 0.000 | 13.1 |
| 4234 | True | 0.900 | 1.122 | 0.000 | 8.1 |
| 5234 | True | 0.900 | 1.118 | 0.000 | 7.0 |
| 6234 | True | 0.900 | 1.112 | 0.000 | 7.6 |
| 7234 | True | 0.900 | 1.093 | 0.000 | 12.3 |
| 8234 | True | 0.900 | 1.102 | 0.000 | 8.1 |
| 9234 | True | 0.900 | 1.129 | 0.000 | 10.0 |
| 10234 | True | 0.900 | 1.113 | 0.000 | 8.3 |
