# Calibration: mountaincar-tilecoding v3

**ADMITTED**

| Criterion | Value |
|---|---|
| Seeds | 10 |
| Reference pass rate | 100% (required ≥ 90%) |
| Random policy fails on every seed | True |
| Deterministic (same seed twice) | True |
| Score unit | mean return |
| Threshold (task) | -110 |
| Gymnasium `reward_threshold` (canonical env) | -110.0 |
| Reference score min / median / max | -106.830 / -100.695 / -99.100 |
| Slowest reference re-run | 4.8 s |
| Suggested train timeout (3×, rounded) | 30 s |

| Seed | Pass | Reward | Score | Random | Re-run s |
|---|---|---|---|---|---|
| 1234 | True | 0.900 | -99.710 | -200.000 | 4.8 |
| 2234 | True | 0.900 | -102.720 | -200.000 | 4.8 |
| 3234 | True | 0.900 | -106.830 | -200.000 | 4.4 |
| 4234 | True | 0.900 | -100.920 | -200.000 | 4.4 |
| 5234 | True | 0.900 | -100.640 | -200.000 | 4.4 |
| 6234 | True | 0.900 | -99.980 | -200.000 | 4.5 |
| 7234 | True | 0.900 | -100.270 | -200.000 | 4.5 |
| 8234 | True | 0.900 | -100.750 | -200.000 | 4.5 |
| 9234 | True | 0.900 | -99.100 | -200.000 | 4.4 |
| 10234 | True | 0.900 | -106.330 | -200.000 | 4.4 |
