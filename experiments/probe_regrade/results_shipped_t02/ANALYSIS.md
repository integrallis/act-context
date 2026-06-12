# exp-015 de-circularized probe re-grade (n=45 probes x 4 arms; held-out repos; independent judge)

## Judge: openai (independent family)
| arm | correct | accuracy | Wilson 95% CI | mean repr tokens |
|---|---|---|---|---|
| source | 30/45 | 0.667 | [0.52, 0.79] | 1123.7 |
| summary_qwen3b | 3/45 | 0.067 | [0.02, 0.18] | 180.0 |
| summary_frontier | 3/45 | 0.067 | [0.02, 0.18] | 164.7 |
| signature | 6/45 | 0.133 | [0.06, 0.26] | 212.1 |

## Judge: claude (same family as answerer — agreement check only)
| arm | correct | accuracy | Wilson 95% CI | mean repr tokens |
|---|---|---|---|---|
| source | 30/45 | 0.667 | [0.52, 0.79] | 1123.7 |
| summary_qwen3b | 2/45 | 0.044 | [0.01, 0.15] | 180.0 |
| summary_frontier | 3/45 | 0.067 | [0.02, 0.18] | 164.7 |
| signature | 4/45 | 0.089 | [0.04, 0.21] | 212.1 |

## McNemar vs source (independent judge)

| arm | arm-only | source-only | p |
|---|---|---|---|
| summary_qwen3b | 0 | 27 | 0.0000 |
| summary_frontier | 0 | 27 | 0.0000 |
| signature | 0 | 24 | 0.0000 |

**Inter-judge agreement:** 0.961 raw; Cohen's kappa = 0.889 (openai vs claude, 180 gradings).
