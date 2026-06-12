# exp-015 de-circularized probe re-grade (n=45 probes x 4 arms; held-out repos; independent judge)

## Judge: openai (independent family)
| arm | correct | accuracy | Wilson 95% CI | mean repr tokens |
|---|---|---|---|---|
| source | 27/45 | 0.600 | [0.45, 0.73] | 1123.7 |
| summary_qwen3b | 4/45 | 0.089 | [0.04, 0.21] | 180.1 |
| summary_frontier | 4/45 | 0.089 | [0.04, 0.21] | 164.9 |
| signature | 6/45 | 0.133 | [0.06, 0.26] | 212.1 |

## Judge: claude (same family as answerer — agreement check only)
| arm | correct | accuracy | Wilson 95% CI | mean repr tokens |
|---|---|---|---|---|
| source | 31/45 | 0.689 | [0.54, 0.80] | 1123.7 |
| summary_qwen3b | 2/45 | 0.044 | [0.01, 0.15] | 180.1 |
| summary_frontier | 2/45 | 0.044 | [0.01, 0.15] | 164.9 |
| signature | 4/45 | 0.089 | [0.04, 0.21] | 212.1 |

## McNemar vs source (independent judge)

| arm | arm-only | source-only | p |
|---|---|---|---|
| summary_qwen3b | 0 | 23 | 0.0000 |
| summary_frontier | 0 | 23 | 0.0000 |
| signature | 0 | 21 | 0.0000 |

**Inter-judge agreement:** 0.944 raw; Cohen's kappa = 0.839 (openai vs claude, 180 gradings).
