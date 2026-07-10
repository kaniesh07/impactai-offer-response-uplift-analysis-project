# Part 2 — Supervised ML: Regression + Classification 

Operates on `data/cleaned_data.csv` from Part 1. Run: `python3 part2/part2_regression_classification.py`
(from `part2/`). 

## Label Definitions

- `y_reg` = `total_spend` (continuous)
- `y_clf` = `offer_completed` (binary: 1 if the customer completed their primary offer, 0 otherwise).
  Chosen instead of median-binarizing `total_spend` (spend is zero-heavy, so a median split would be
  degenerate) — `offer_completed` is a natural, pre-existing binary business outcome, per the
  rubric's explicit fallback clause.

6 rows with no primary offer (customers who never received any offer at all, so `offer_type`/etc.
are undefined) were dropped, leaving **16,994 rows**.

## ⚠️ Data Leakage Found and Fixed

Before finalizing this Part, a leakage check was run: `n_offers_completed == 0` implies
`offer_completed == 0` **with 100% certainty** across all 16,994 rows . This makes sense
mechanically — a customer who completed zero offers total cannot have completed their *primary*
offer — but it means `n_offers_completed` (and, for the same reason, `avg_time_to_complete_hours`
and the `has_completed_offer` flag) **leak the classification target almost perfectly**, which would
have produced an artificially inflated AUC (an earlier run before this fix scored AUC = 0.989 —
suspiciously close to perfect).

**Fix:** these three columns are excluded from the classification feature set (`X_clf`, 19 columns)
but retained for the regression feature set (`X_reg`, 20 columns), since `total_spend` is not
mechanically derived from offer-completion counts — that relationship there is a legitimate signal,
not leakage. Two separate `StandardScaler` instances are fit accordingly (one per feature set), each
strictly on its own training split.

## Task 1 — Encoding

- **Ordinal**: `income_bracket` → {Low:0, Medium:1, High:2}, justified because income level is a
  genuinely ranked quantity.
- **Nominal (one-hot)**: `gender`, `offer_type`, and `channels_*` binary flags — label encoding here
  would falsely imply an order (e.g., bogo < discount < informational has no real meaning a linear
  model could correctly exploit).

## Task 2 — Leak-Free Split + Scaling

80/20 `train_test_split(random_state=42)`. `StandardScaler` is fit **only** on each training split
and applied to test via `.transform()`, never `.fit_transform()` on test data — fitting on the full
dataset would leak test-set mean/variance into training, producing an overly optimistic performance
estimate.

## Task 3 — Linear Regression (predict `total_spend`)

**MSE: 9,682.44 | R²: 0.406**

Top-3 coefficients by magnitude: `n_offers_completed` (47.74), `n_transactions` (37.02), `income`
(30.26) — these are the levers with the largest predicted swing in spend per standard deviation of
change, and are legitimate (non-leaky) drivers for the regression task.

## Task 4 — Ridge Regression (alpha=1.0)

**MSE: 9,682.42 | R²: 0.406** — nearly identical to OLS at this alpha, with the largest coefficient
shrinkage on `difficulty` and `income` (features correlated with others like `duration`/`reward`).
Ridge trades negligible training fit for a more stable coefficient profile under multicollinearity.

## Task 5 — Class Imbalance Check

`offer_completed` distribution: 65.8% positive / **34.2% negative** (minority class). Since the
minority class share (34.2%) falls just under the 35% threshold, `class_weight='balanced'` **was
applied** to the Logistic Regression model.

## Logistic Regression Results (leak-free features)

**Confusion matrix:**
```
[[ 963  210]
 [ 311 1915]]
```
**Accuracy: 0.847 | Precision: 0.901 | Recall: 0.860 | F1: 0.880 | AUC: 0.921**

ROC curve saved to `plots/roc_curve_logreg.png`.

## Business Framing: Precision vs Recall

- **Precision** = of customers predicted to complete an offer, what fraction actually do.
- **Recall** = of customers who actually complete an offer, what fraction are correctly identified.
- **False negatives** = missed marketing opportunity — a customer who would have responded is never
  targeted, forfeiting revenue.
- **False positives** = wasted incentive cost — a reward-bearing offer sent to someone who was never
  going to redeem it.

Given that the value of reaching a genuine completer typically exceeds the marginal cost of one
unredeemed offer, this business context slightly favors **recall over precision**.

## Task 5b — Threshold Sensitivity (0.30–0.70)

| threshold | precision | recall | f1 |
|---|---|---|---|
| 0.30 | 0.858 | 0.947 | **0.901** |
| 0.40 | 0.881 | 0.911 | 0.896 |
| 0.50 | 0.901 | 0.860 | 0.880 |
| 0.60 | 0.918 | 0.796 | 0.852 |
| 0.70 | 0.933 | 0.710 | 0.807 |

**F1-maximizing threshold: 0.30** (F1 = 0.901). Consistent with the recall-favoring business framing
above, this lower threshold is a reasonable operating point — it avoids over-correcting toward
precision at the cost of missing likely completers.

## Task 6 — Regularization Experiment (C=0.01 vs C=1.0)

| C | accuracy | precision | recall | f1 | auc | mean\|coef\| |
|---|---|---|---|---|---|---|
| 0.01 | 0.843 | 0.898 | 0.858 | 0.878 | 0.919 | 0.370 |
| 1.00 | 0.847 | 0.901 | 0.860 | 0.880 | 0.921 | 0.536 |

`C` is the inverse regularization strength in scikit-learn (smaller `C` = stronger L2 penalty).
`C=0.01` shrinks coefficients substantially (lower mean\|coef\|), producing a simpler, more
conservative model; `C=1.0` fits the training data more closely, with a modest overfitting risk.

## Task 7 — Bootstrap AUC 95% CI (500 resamples)

- Mean bootstrap AUC: **0.9209**
- Mean AUC difference from random baseline (0.5): **0.4209**
- **95% CI: [0.9107, 0.9299]**
- The CI **excludes 0.5** → the model performs significantly better than random guessing.
