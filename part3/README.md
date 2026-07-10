# Part 3 — Ensembles, Tuning, Full ML Pipeline 

Operates on `data/cleaned_data.csv`, reusing the **same leak-free classification feature set** from
Part 2 (excludes `n_offers_completed`, `avg_time_to_complete_hours`, `has_completed_offer` — see
Part 2 README for the leakage finding). Target: `offer_completed`. Run:
`python3 part3/part3_ensembles_pipeline.py` (from `part3/`). 

## Unconstrained vs Controlled Decision Tree

| Tree | Train acc | Test acc | Gap |
|---|---|---|---|
| Unconstrained | 1.000 | 0.829 | **0.171** |
| Controlled (max_depth=5, min_samples_split=20) | 0.836 | 0.841 | **-0.005** |

The unconstrained tree grows until leaves are pure, memorizing training-set noise — the large gap is
classic high-variance overfitting. Capping depth and requiring a minimum split size sharply reduces
that gap (even slightly favoring test performance here), trading a little training accuracy for
much better generalization.

## Gini vs Entropy (both max_depth=5)

- Gini test accuracy: **0.8411**
- Entropy test accuracy: **0.8361**
- Gini impurity: `Gini(t) = 1 - Σ p_i²` — probability of misclassifying a random element if labeled
  by the node's class distribution.
- Entropy: `H(t) = -Σ p_i·log₂(p_i)` — information/disorder at a node in bits.

Both criteria typically produce similar splits in practice; entropy is marginally more expensive
computationally due to the logarithm.

## Random Forest (n_estimators=100, max_depth=10)

Train accuracy: 0.907 | **Test accuracy: 0.884 | Test ROC-AUC: 0.949**

**Top-5 feature importances:** `n_transactions` (0.183), `reward` (0.131), `duration` (0.116),
`difficulty` (0.104), `membership_tenure_days` (0.093). Offer-economics features (reward, duration,
difficulty) rank alongside behavioral/demographic ones — a substantively interesting result: *how*
an offer is designed matters roughly as much as *who* the customer is.

**Bagging explanation:** each tree trains on a bootstrap resample of training rows and considers
only a random ~√n_features subset of candidates at each split. This decorrelates individual trees —
no single feature can dominate every tree's early splits — so averaging reduces variance far more
than one deep tree could alone, which is why several distinct features (not just one) surface as
important above.

## Gradient Boosting (n_estimators=100, learning_rate=0.1, max_depth=3)

Test accuracy: 0.887 | **Test ROC-AUC: 0.951**

## Feature Ablation (drop 5 lowest-importance features)

Dropped: `channel_web`, `channel_social`, `channel_mobile`, `gender_O`, `channel_email` (importance
**0.000000** — verified this is because every offer in the dataset includes the email channel, so
`channel_email` is a constant column with zero information content, not a bug).

Full model AUC: 0.9493 → Reduced model (14 features) AUC: 0.9494 (**+0.0001**, i.e. no real change).
**Production trade-off:** a negligible AUC cost for a meaningfully simpler model is worth it at
marketing scale — fewer features means cheaper feature computation/storage per scoring pass, easier
drift monitoring, and lower-latency inference when scoring millions of customers ahead of a campaign.

## 5-Fold Stratified CV Comparison (scoring=roc_auc)

| Model | CV mean AUC | CV std AUC |
|---|---|---|
| LogisticRegression | 0.9194 | 0.0069 |
| DecisionTree (ctrl, depth=5) | 0.9020 | 0.0082 |
| RandomForest | 0.9453 | 0.0047 |
| GradientBoosting | 0.9473 | 0.0047 |

## GridSearchCV (Imputer → Scaler → RandomForestClassifier)

Grid: `n_estimators ∈ {50,100,150}`, `max_depth ∈ {5,10,15}`, `min_samples_leaf ∈ {1,5,10}` →
**27 configurations × 5 folds = 135 total fits**.

**Best params:** `max_depth=15, min_samples_leaf=5, n_estimators=150` | **Best CV AUC: 0.9470** |
**Test AUC: 0.9510**

**Grid vs Randomized Search:** GridSearchCV exhaustively evaluates every configuration, guaranteeing
the in-grid optimum but scaling multiplicatively with each added hyperparameter/value.
RandomizedSearchCV instead samples a fixed budget of random configurations from the same space —
for large grids this typically finds a near-optimal configuration for a fraction of the compute, at
the cost of no formal optimality guarantee. At this grid's size (27 configs), exhaustive search is
still cheap enough to be the better choice; it stops being practical once hyperparameters have many
more values or more hyperparameters are added.

## Manual Learning Curve (best pipeline, 20/40/60/80/100% of training data)

| % data | n samples | Train AUC | Test AUC |
|---|---|---|---|
| 20% | 2,719 | 0.9809 | 0.9437 |
| 40% | 5,438 | 0.9798 | 0.9481 |
| 60% | 8,157 | 0.9803 | 0.9499 |
| 80% | 10,876 | 0.9801 | 0.9493 |
| 100% | 13,595 | 0.9799 | 0.9510 |

Train/test gap at 100% data: 0.029. Test AUC improvement from 20%→100% of data: +0.007.
**Conclusion: roughly converged** — test AUC has plateaued and the train/test gap is stable,
suggesting the model has extracted most of the signal available in this feature set at its current
capacity rather than being starved for more data. Plot: `plots/learning_curve.png`.

## Model Serialization

`best_model.pkl` (the GridSearchCV-winning pipeline) saved via `joblib.dump`. Reload-and-predict
sanity check on 2 hand-crafted profiles:

- **Profile A** (high-income, highly engaged): predicted `offer_completed=1`, probability = 0.9864
- **Profile B** (low-engagement, informational offer, never viewed): predicted `offer_completed=0`,
  probability = 0.0000

Both predictions are directionally sensible given the profiles' construction.

## Summary Comparison Table (Parts 2 + 3)

| Model | CV mean AUC | CV std AUC | Test AUC |
|---|---|---|---|
| LogisticRegression | 0.9194 | 0.0069 | 0.9302 |
| DecisionTree (ctrl, depth=5) | 0.9020 | 0.0082 | 0.9093 |
| RandomForest (default) | 0.9453 | 0.0047 | 0.9493 |
| GradientBoosting | 0.9473 | 0.0047 | **0.9511** |
| **RandomForest (GridSearchCV-tuned)** | 0.9470 | — | 0.9510 |

**Final recommendation: RandomForest (GridSearchCV-tuned).** GradientBoosting scores marginally
higher on this single test split (0.9511 vs 0.9510), but the 0.0001 gap is well within the ~0.005 CV
standard deviation observed across models — not a meaningful difference. The tuned RandomForest is
preferred for production because it was formally selected via 5-fold GridSearchCV on the training
set only (test data untouched during selection, an honest non-leaked estimate) inside a complete
`Imputer → Scaler → Classifier` pipeline, making it directly deployable end-to-end. This is the model
serialized as `best_model.pkl` and carried into Part 4.
