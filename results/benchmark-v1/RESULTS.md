# Routing benchmark results

Frozen weights; feedback updates state only. CPU NumPy; synthetic task.

| Condition | Controller | Accuracy | Brier | Log loss | ECE | Confidently wrong | Sampled accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| standard | fixed_expert | 49.661% | 0.4555 | 1.5335 | 0.4534 | 50.339% | 49.661% |
| standard | fixed_uniform | 51.615% | 0.2500 | 0.6931 | 0.0161 | 0.000% | 49.219% |
| standard | recency | 88.281% | 0.0990 | 0.3427 | 0.0302 | 4.609% | 82.995% |
| standard | autoregressive | 87.778% | 0.1004 | 0.3468 | 0.0219 | 5.286% | 84.123% |
| standard | recurrent | 89.149% | 0.0917 | 0.3241 | 0.0218 | 5.425% | 85.365% |
| fast_switch | fixed_expert | 51.042% | 0.4431 | 1.4928 | 0.4396 | 48.958% | 51.042% |
| fast_switch | fixed_uniform | 49.922% | 0.2500 | 0.6931 | 0.0195 | 0.000% | 50.182% |
| fast_switch | recency | 81.615% | 0.1431 | 0.4621 | 0.0526 | 6.458% | 77.370% |
| fast_switch | autoregressive | 80.825% | 0.1498 | 0.4815 | 0.0655 | 7.752% | 77.951% |
| fast_switch | recurrent | 78.585% | 0.1625 | 0.5153 | 0.0888 | 8.594% | 76.727% |
| random_labels | fixed_expert | 49.505% | 0.4570 | 1.5381 | 0.4549 | 50.495% | 49.505% |
| random_labels | fixed_uniform | 52.604% | 0.2500 | 0.6931 | 0.0260 | 0.000% | 50.885% |
| random_labels | recency | 50.026% | 0.3495 | 1.0127 | 0.2903 | 14.453% | 51.042% |
| random_labels | autoregressive | 50.260% | 0.3628 | 1.0687 | 0.3122 | 19.280% | 50.269% |
| random_labels | recurrent | 50.373% | 0.3617 | 1.0630 | 0.3076 | 18.637% | 50.677% |
| no_feedback | fixed_expert | 49.661% | 0.4555 | 1.5335 | 0.4534 | 50.339% | 49.661% |
| no_feedback | fixed_uniform | 51.615% | 0.2500 | 0.6931 | 0.0161 | 0.000% | 49.219% |
| no_feedback | recency | 51.615% | 0.2500 | 0.6931 | 0.0161 | 0.000% | 49.219% |
| no_feedback | autoregressive | 49.661% | 0.2585 | 0.7106 | 0.0881 | 0.000% | 51.302% |
| no_feedback | recurrent | 50.217% | 0.3210 | 0.8881 | 0.2400 | 0.104% | 49.896% |

## Preregistered scientific gates

- learning: PASS
- negative_control: PASS
- memory_dependence: PASS
- comparative: FAIL

AR minus recurrent accuracy by independent test episode seed: [-0.008246527777777679, -0.015624999999999889, -0.01736111111111105, -0.01822916666666652, -0.00911458333333337]

Three parameter seeds share five test streams; 15 runs are not 15 independent environments.
No hardware calibration, quantum advantage, AGI, or superiority outside this task is established.
The expert bank and feedback features are hand designed and shared by all adaptive baselines.
