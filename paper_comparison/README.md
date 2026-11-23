# SentinelNet published-baseline comparison

Reference: B. Ingre, A. Yadav and A. K. Soni (2017), *Decision Tree Based Intrusion Detection System for NSL-KDD Dataset*, ICTIS 2017, Springer, pp. 207–218. DOI: https://doi.org/10.1007/978-3-319-63645-0_23. Author-posted full text: https://www.researchgate.net/publication/315797238_Decision_Tree_Based_Intrusion_Detection_System_for_NSL-KDD_Dataset

## Repeat the experiment

Use the existing SentinelNet Python environment; requires numpy, pandas, scikit-learn. Copy `compare_published_baseline.py` into the SentinelNet project directory, open CMD in that directory, and run:

```cmd
python compare_published_baseline.py --data data/nsl_kdd
```

The script never edits or loads the application model cache. Its output folder contains measured results, validation-based pruning selection, test confusion matrices and file hashes. It retrains the same Basic and Enhanced Random Forest configurations (100 trees, seed 42) used by SentinelNet. The run included here used scikit-learn 1.8.0; small numerical differences may occur with other versions.

## Experimental scope

This is an adaptation of the reference's 14-feature, Gini CART method, not an exact replication of its MATLAB/WEKA implementation, label scope, sample selection or published accuracy. Both models were trained on the same 124,926 Normal/DoS/Probe KDDTrain+ records and evaluated on the identical 19,592 KDDTest+ records. The reference's fixed 14-feature list is used directly; CART pruning was selected solely using an 80/20 stratified training validation split. The two excluded attack classes R2L and U2R are not evaluated. Categorical values are frequency-ranked from training only.

Under simulated missingness, the same randomly chosen records retain only protocol, duration, source bytes and destination bytes. The fixed CART must impute its ten missing features from training statistics (numeric medians, categorical modes). Adaptive SentinelNet instead uses its four-feature RF for those records, and the ten-feature RF for complete records. Five fixed seeds (42–46) were used at each missingness level. This imputation policy is an explicitly defined local comparison, not a missing-data method attributed to the paper.

## Findings

| Basic-only records | Adaptive SentinelNet accuracy | Reference-inspired CART accuracy |
|---|---:|---:|
| 0% | 85.16% | 88.43% |
| 25% | 84.31% | 80.80% |
| 50% | 83.46% | 73.31% |
| 75% | 82.69% | 65.78% |
| 100% | 81.88% | 58.19% |

At the 50% scenario, mean accuracy improves by 10.15 percentage points and macro F1 improves from 0.6877 to 0.7632, but Probe recall falls from 49.77% to 45.00%. Complete-data CART is more accurate. SentinelNet uses 10 features instead of 14 (28.57% fewer), and retains classification with only four fields. The supported conclusion is increased robustness under this simulated missing-feature setting, not universal superiority, modern-network generalization, or proof of globally novel algorithms. The benchmark does not address five-class or live traffic performance.

The existing project does not need to be modified to run this comparison.
