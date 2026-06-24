# Factor Effectiveness Validation

This report validates the `new_all1244` universe on `2020-01` using the two model-specific tests requested:

- Ridge: exact leave-one-factor-out retraining. A factor is retained only when removing it lowers `2020-01` OOS `pred_xsz` IC.
- LightGBM: single-factor standard-normal replacement on `2020-01`. A factor is retained only when replacing it lowers OOS `pred_xsz` IC.

## Summary

| model | validation | base_ic_2020_01 | retained | removed | retained_from_new100 | removed_from_new100 |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge | leave-one-factor-out retrain, remove if IC does not decline | 0.05639702 | 617 | 627 | 51 | 49 |
| LightGBM | single-factor standard-normal replacement, remove if IC does not decline | 0.03305025 | 643 | 601 | 46 | 54 |

Among the 100 newly mined expression factors, Ridge retains 51 and LightGBM retains 46. Their retained-new-factor overlap is 26.

## Retained New-Factor Overlap

| factor |
| --- |
| nf_rank_add_1c820a8206 |
| nf_rank_add_253ba09fcd |
| nf_rank_add_25b378dbdb |
| nf_rank_add_2738141b2a |
| nf_rank_add_5b6135e7ba |
| nf_rank_add_61da7be3bb |
| nf_rank_add_62dc7f09de |
| nf_rank_add_7d4b606c47 |
| nf_rank_add_7f5ba929ad |
| nf_rank_add_8ac76f15c2 |
| nf_rank_spread_67780b112d |
| nf_z_add_018853eb7e |
| nf_z_add_1670531f56 |
| nf_z_add_2020072977 |
| nf_z_add_453c606cbe |
| nf_z_add_6f65c98834 |
| nf_z_add_73c01cd34a |
| nf_z_add_8c93a2237b |
| nf_z_add_901d45b9c1 |
| nf_z_add_a1f2a113eb |
| nf_z_add_b80bcddc95 |
| nf_z_add_c56c57132b |
| nf_z_add_e0a0f73e07 |
| nf_z_add_e1fa23c467 |
| nf_z_spread_1a3bbef3f2 |
| nf_z_spread_415253b244 |

## Strongest New Factors Under Ridge Test

| factor | drop_ic | delta_ic | retained |
| --- | --- | --- | --- |
| nf_rank_add_7d4b606c47 | 0.05624507 | 0.00015195 | True |
| nf_z_add_8c93a2237b | 0.05628916 | 0.00010786 | True |
| nf_z_add_2020072977 | 0.05636287 | 0.00003415 | True |
| nf_z_add_a1f2a113eb | 0.05636287 | 0.00003415 | True |
| nf_z_add_0d8e6b35c5 | 0.05637063 | 0.00002639 | True |
| nf_z_add_c2a94707f7 | 0.05637081 | 0.00002622 | True |
| nf_rank_add_163c6a8662 | 0.05637765 | 0.00001937 | True |
| nf_rank_add_4845115b14 | 0.05637792 | 0.00001910 | True |
| nf_z_add_d0643a432f | 0.05638306 | 0.00001396 | True |
| nf_rank_add_02b3b1afc9 | 0.05638412 | 0.00001290 | True |
| nf_z_add_6f65c98834 | 0.05638496 | 0.00001206 | True |
| nf_rank_add_a449929f91 | 0.05638541 | 0.00001161 | True |
| nf_rank_add_e0e28ff6f6 | 0.05638578 | 0.00001125 | True |
| nf_z_spread_415253b244 | 0.05638841 | 0.00000861 | True |
| nf_z_spread_1a3bbef3f2 | 0.05638841 | 0.00000861 | True |

## Strongest New Factors Under LightGBM Shuffle Test

| factor | shuffled_ic | delta_ic | retained | split_importance |
| --- | --- | --- | --- | --- |
| nf_rank_add_80760062dd | 0.03132456 | 0.00172569 | True | 6 |
| nf_rank_add_7c6d8cf169 | 0.03161339 | 0.00143686 | True | 8 |
| nf_rank_add_7d4b606c47 | 0.03177522 | 0.00127503 | True | 11 |
| nf_rank_add_c2f4d07ecb | 0.03186727 | 0.00118298 | True | 8 |
| nf_rank_add_91628c8cfc | 0.03201976 | 0.00103049 | True | 6 |
| nf_z_add_bd94dd0e65 | 0.03217728 | 0.00087296 | True | 3 |
| nf_rank_add_a8f0f2c5fc | 0.03245287 | 0.00059738 | True | 4 |
| nf_rank_add_7e2af87fba | 0.03247722 | 0.00057303 | True | 3 |
| nf_z_add_20e37afa07 | 0.03248144 | 0.00056881 | True | 8 |
| nf_z_add_28054e553a | 0.03269949 | 0.00035076 | True | 5 |
| nf_z_add_018853eb7e | 0.03274940 | 0.00030085 | True | 3 |
| nf_rank_add_2738141b2a | 0.03276414 | 0.00028611 | True | 4 |
| nf_z_add_1670531f56 | 0.03277001 | 0.00028024 | True | 5 |
| nf_z_add_b63ca85c96 | 0.03279621 | 0.00025404 | True | 4 |
| nf_z_add_8c93a2237b | 0.03280846 | 0.00024179 | True | 5 |

## Artifacts

- Ridge full validation CSV: `reports/generated/effectiveness_validation/ridge_leave_one_2020-01.csv`
- LightGBM full validation CSV: `reports/generated/effectiveness_validation/lgbm_shuffle_2020-01.csv`
- Combined summary JSON: `reports/generated/effectiveness_validation/factor_effectiveness_2020-01_summary.json`
- Ridge retained list: `reports/generated/effectiveness_validation/ridge_retained_2020-01.txt`
- LightGBM retained list: `reports/generated/effectiveness_validation/lgbm_retained_2020-01.txt`
