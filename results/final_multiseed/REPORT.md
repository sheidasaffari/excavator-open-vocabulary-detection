# Selected 100K curated and random configurations across three seeds

Configurations were selected and frozen on MOCS before held-out evaluation. Values are mean ± sample standard deviation across seeds 0, 1, and 2. The prompt is `construction excavator`; the prediction floor is 0.001; the operating threshold is 0.25.

## Positive datasets

| Family | Dataset | AP50 | AP50:95 | P@0.25 | R@0.25 | F1@0.25 |
|---|---|---:|---:|---:|---:|---:|
| Curated | mocs_dev | 0.776 ± 0.006 | 0.585 ± 0.003 | 0.955 ± 0.024 | 0.405 ± 0.173 | 0.552 ± 0.178 |
| Curated | cis_test | 0.870 ± 0.007 | 0.670 ± 0.005 | 0.931 ± 0.025 | 0.534 ± 0.229 | 0.655 ± 0.199 |
| Curated | positive_control | 0.609 ± 0.004 | 0.277 ± 0.003 | 0.789 ± 0.062 | 0.345 ± 0.173 | 0.457 ± 0.173 |
| Random | mocs_dev | 0.768 ± 0.002 | 0.578 ± 0.007 | 0.946 ± 0.022 | 0.477 ± 0.084 | 0.630 ± 0.068 |
| Random | cis_test | 0.857 ± 0.008 | 0.663 ± 0.014 | 0.918 ± 0.045 | 0.625 ± 0.068 | 0.740 ± 0.032 |
| Random | positive_control | 0.571 ± 0.013 | 0.255 ± 0.009 | 0.747 ± 0.052 | 0.397 ± 0.087 | 0.512 ± 0.068 |

## Corrected confuser set

| Family | Image FPR | False predictions per 100 images |
|---|---:|---:|
| Curated | 0.142 ± 0.119 | 14.5 ± 11.9 |
| Random | 0.173 ± 0.081 | 19.1 ± 8.7 |
