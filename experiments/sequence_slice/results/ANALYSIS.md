# exp-014c sequence-slice vs keep/drop (N=70; 0 eval-incomplete excluded)

| arm | resolved | rate | Wilson 95% CI | mean ctx tok |
|---|---|---|---|---|
| oracle_seq | 26/70 | 0.371 | [0.27, 0.49] | 7091.9 |
| oracle_keep_drop | 23/70 | 0.329 | [0.23, 0.44] | 6876.0 |

**H3 (seq > keep/drop), exact McNemar:** seq-only 4, kd-only 1, p = 0.3750

Instances with a non-empty slice: 65/70; mean slice size 215.5 tok.

Stability: keep/drop re-run resolves 23 vs exp-014b's 25 on the same instances; symmetric difference 6 (model/temporal drift indicator).
