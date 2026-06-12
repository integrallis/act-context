# exp-014b v2 analysis (N=70; excluded: 1 gold-validation, 0 eval-incomplete)

## All evaluated instances (n=70)

| arm | resolved | rate | Wilson 95% CI | mean cl100k ctx tok | mean coverage |
|---|---|---|---|---|---|
| oracle_full | 19/70 | 0.271 | [0.18, 0.39] | 25426.3 | 1.0 |
| oracle_tier | 23/70 | 0.329 | [0.23, 0.44] | 8629.1 | 1.0 |
| oracle_keep_drop | 25/70 | 0.357 | [0.26, 0.47] | 6876.0 | 1.0 |

### McNemar (exact, two-sided)

| A vs B | A-only | B-only | discordant | p |
|---|---|---|---|---|
| oracle_tier vs oracle_keep_drop | 4 | 6 | 10 | 0.754 |
| oracle_tier vs oracle_full | 7 | 3 | 10 | 0.344 |
| oracle_keep_drop vs oracle_full | 7 | 1 | 8 | 0.070 |

## Expressible subset (every arm covers every gold hunk pre-image) (n=70)

| arm | resolved | rate | Wilson 95% CI | mean cl100k ctx tok | mean coverage |
|---|---|---|---|---|---|
| oracle_full | 19/70 | 0.271 | [0.18, 0.39] | 25426.3 | 1.0 |
| oracle_tier | 23/70 | 0.329 | [0.23, 0.44] | 8629.1 | 1.0 |
| oracle_keep_drop | 25/70 | 0.357 | [0.26, 0.47] | 6876.0 | 1.0 |

### McNemar (exact, two-sided)

| A vs B | A-only | B-only | discordant | p |
|---|---|---|---|---|
| oracle_tier vs oracle_keep_drop | 4 | 6 | 10 | 0.754 |
| oracle_tier vs oracle_full | 7 | 3 | 10 | 0.344 |
| oracle_keep_drop vs oracle_full | 7 | 1 | 8 | 0.070 |

