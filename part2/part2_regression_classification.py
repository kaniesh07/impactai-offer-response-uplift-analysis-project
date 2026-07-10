"""
Part 2 -- Supervised ML: Regression + Classification 

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, roc_auc_score,
)

os.makedirs("plots", exist_ok=True)
np.random.seed(42)

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


df = pd.read_csv("../data/cleaned_data.csv")

# Drop the 6 rows with no primary offer at all (customers who never received
# any offer -- offer_type/reward/etc. are NaN and offer_completed is
# undefined for them by construction).
before = len(df)
df = df.dropna(subset=["offer_type"]).reset_index(drop=True)
log(f"Dropped {before - len(df)} rows with no primary offer (undefined offer_type). "
    f"Remaining: {len(df)} rows.")


# Label definitions

log("\n" + "=" * 70)
log("LABEL DEFINITIONS")
log("=" * 70)
log("y_reg = total_spend (continuous)")
log("y_clf = offer_completed (binary: 1 if the customer completed their "
    "primary offer within its validity window, 0 otherwise). Chosen "
    "instead of median-binarizing total_spend, since spend is zero-heavy "
    "and a median split would be degenerate; offer_completed is a "
    "natural, pre-existing binary business outcome.")

y_reg = df["total_spend"]
y_clf = df["offer_completed"]


# Task 1: Encoding

log("\n" + "=" * 70)
log("TASK 1: Encoding")
log("=" * 70)

# Ordinal: income_bracket (Low < Medium < High). 'Unknown' rows (originally
# missing income) are mapped to the imputed-income tertile they now fall
# into, since income itself was already median-imputed in Part 1 -- so we
# recompute the bracket from the (now fully populated) income column for
# modeling consistency rather than leaving an ordinal 'Unknown' bucket.
q1, q2 = df["income"].quantile([1 / 3, 2 / 3])
def bracket_ordinal(x):
    if x <= q1:
        return 0  # Low
    elif x <= q2:
        return 1  # Medium
    else:
        return 2  # High
df["income_bracket_encoded"] = df["income"].apply(bracket_ordinal)
log("Ordinal encoding: income_bracket -> {Low:0, Medium:1, High:2}. Ordering "
    "is justified because these categories represent a genuine ranked "
    "quantity (income level), so label encoding preserves meaningful "
    "distance/order information a model can use directly.")

# Nominal: gender, offer_type, channels -> one-hot
log("\nNominal (one-hot) encoding: gender, offer_type, channels_* flags. "
    "Label encoding would be inappropriate here because it would falsely "
    "imply an order (e.g., bogo=0 < discount=1 < informational=2 has no "
    "real meaning -- discount is not 'more' than bogo in any sense a "
    "linear model could exploit correctly).")

df["channels"] = df["channels"].apply(eval)  # stored as string repr of list
for ch in ["web", "email", "mobile", "social"]:
    df[f"channel_{ch}"] = df["channels"].apply(lambda lst: int(ch in lst))

gender_dummies = pd.get_dummies(df["gender"], prefix="gender", drop_first=True)
offer_dummies = pd.get_dummies(df["offer_type"], prefix="offer", drop_first=True)

# NOTE on leakage: n_offers_completed, avg_time_to_complete_hours, and
# has_completed_offer are all direct functions of whether/how a customer
# completed an offer. Verification showed n_offers_completed == 0 implies
# offer_completed == 0 with 100% certainty (a customer with zero total
# offer completions cannot have completed their primary offer) -- so these
# columns leak the classification target almost perfectly and are EXCLUDED
# from the classification feature set below. They remain valid features for
# the regression task (predicting total_spend), since total_spend is not
# mechanically derived from offer-completion counts.
LEAKY_FOR_CLF = ["n_offers_completed", "avg_time_to_complete_hours", "has_completed_offer"]

feature_cols_numeric = [
    "age", "income", "membership_tenure_days", "income_bracket_encoded",
    "n_transactions", "n_offers_received", "n_offers_viewed",
    "n_offers_completed", "reward", "difficulty", "duration",
    "channel_web", "channel_email", "channel_mobile", "channel_social",
]
X_reg = pd.concat([df[feature_cols_numeric], gender_dummies, offer_dummies], axis=1).astype(float)

feature_cols_clf = [c for c in feature_cols_numeric if c not in LEAKY_FOR_CLF]
X_clf = pd.concat([df[feature_cols_clf], gender_dummies, offer_dummies], axis=1).astype(float)

X = X_reg  # used for the regression tasks below (Tasks 3-4)
log(f"\nRegression feature matrix shape: {X_reg.shape}")
log(f"Regression feature columns: {list(X_reg.columns)}")
log(f"\nClassification feature matrix shape: {X_clf.shape} "
    f"(excludes leaky columns: {LEAKY_FOR_CLF})")
log(f"Classification feature columns: {list(X_clf.columns)}")


# Task 2: Leak-free split + scaling

log("\n" + "=" * 70)
log("TASK 2: Leak-free split + scaling")
log("=" * 70)
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
Xc_train, Xc_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)
log(f"Regression train/test shapes: {X_train.shape} / {X_test.shape}")
log(f"Classification train/test shapes: {Xc_train.shape} / {Xc_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_scaler = StandardScaler()
Xc_train_scaled = clf_scaler.fit_transform(Xc_train)
Xc_test_scaled = clf_scaler.transform(Xc_test)
log("StandardScaler fit ONLY on the respective X_train (a separate scaler "
    "for the regression feature set and one for the classification feature "
    "set, since their columns differ), then applied to X_test via "
    "transform() (not fit_transform). Fitting the scaler on the full "
    "dataset (or on test data) would leak test-set mean/variance "
    "information into the training pipeline, giving an overly optimistic "
    "estimate of generalization performance -- the scaler's parameters "
    "must come exclusively from data the model is 'allowed' to see during "
    "training.")


# Task 3: Linear Regression

log("\n" + "=" * 70)
log("TASK 3: Linear Regression (predict total_spend)")
log("=" * 70)
lr = LinearRegression()
lr.fit(X_train_scaled, y_reg_train)
pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_reg_test, pred_lr)
r2_lr = r2_score(y_reg_test, pred_lr)
log(f"MSE: {mse_lr:.3f}")
log(f"R^2: {r2_lr:.3f}")

coef_series = pd.Series(lr.coef_, index=X.columns).sort_values(key=abs, ascending=False)
log(f"\nTop-3 coefficients by magnitude:\n{coef_series.head(3)}")
log("Interpretation: coefficients with the largest magnitude drive the "
    "biggest swings in predicted spend per standard-deviation change in "
    "that (scaled) feature -- these are the levers most worth targeting "
    "in a marketing strategy.")


# Task 4: Ridge Regression

log("\n" + "=" * 70)
log("TASK 4: Ridge Regression (alpha=1.0)")
log("=" * 70)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_reg_test, pred_ridge)
r2_ridge = r2_score(y_reg_test, pred_ridge)

comparison_df = pd.DataFrame({
    "OLS_coef": lr.coef_, "Ridge_coef": ridge.coef_
}, index=X.columns)
comparison_df["abs_shrinkage"] = (comparison_df["OLS_coef"].abs() - comparison_df["Ridge_coef"].abs())
log(f"OLS  -> MSE: {mse_lr:.3f}, R^2: {r2_lr:.3f}")
log(f"Ridge-> MSE: {mse_ridge:.3f}, R^2: {r2_ridge:.3f}")
log(f"\nCoefficient comparison (largest shrinkage first):\n"
    f"{comparison_df.sort_values('abs_shrinkage', ascending=False).head(5)}")
log("Ridge's L2 penalty shrinks coefficient magnitudes toward zero, "
    "especially for features that are correlated with others (e.g., "
    "difficulty/duration/reward, which move together by offer design) -- "
    "this trades a small amount of training-set fit for a more stable, "
    "less variance-prone coefficient profile.")


# Task 5: Class imbalance check

log("\n" + "=" * 70)
log("TASK 5: Class imbalance check")
log("=" * 70)
class_dist = y_clf.value_counts(normalize=True)
log(f"offer_completed distribution:\n{class_dist}")
minority_pct = class_dist.min() * 100
use_balanced = minority_pct < 35
log(f"Minority class share: {minority_pct:.1f}%")
log(f"Apply class_weight='balanced'? {use_balanced} "
    f"(threshold: minority class < 35%)")
class_weight_arg = "balanced" if use_balanced else None
log(f"class_weight parameter used for Logistic Regression: {class_weight_arg}")


# Task 5b (part of Task 5 block): Logistic Regression

log("\n" + "=" * 70)
log("TASK 5 (cont.): Logistic Regression")
log("=" * 70)
logreg = LogisticRegression(max_iter=1000, class_weight=class_weight_arg, random_state=42)
logreg.fit(Xc_train_scaled, y_clf_train)
pred_clf = logreg.predict(Xc_test_scaled)
pred_proba = logreg.predict_proba(Xc_test_scaled)[:, 1]

cm = confusion_matrix(y_clf_test, pred_clf)
acc = accuracy_score(y_clf_test, pred_clf)
prec = precision_score(y_clf_test, pred_clf)
rec = recall_score(y_clf_test, pred_clf)
f1 = f1_score(y_clf_test, pred_clf)
auc = roc_auc_score(y_clf_test, pred_proba)

log(f"Confusion matrix:\n{cm}")
log(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | "
    f"F1: {f1:.3f} | AUC: {auc:.3f}")

fpr, tpr, _ = roc_curve(y_clf_test, pred_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={auc:.3f})", color="darkred")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve -- offer_completed Classifier")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_curve_logreg.png", dpi=110)
plt.close()
log("Saved ROC curve -> plots/roc_curve_logreg.png")


# Business framing: precision vs recall

log("\n" + "=" * 70)
log("Business framing: Precision vs Recall")
log("=" * 70)
log("Precision = TP / (TP + FP): of customers we PREDICT will complete an "
    "offer, what fraction actually do.")
log("Recall = TP / (TP + FN): of customers who ACTUALLY complete an offer, "
    "what fraction we correctly identify.")
log("False negatives (predicting non-completion for an actual completer) "
    "= a missed marketing opportunity -- we fail to reach a customer who "
    "was ready to respond, forfeiting revenue we could have captured. "
    "False positives (predicting completion for a non-completer) = wasted "
    "incentive cost -- we send a reward-bearing offer to someone who was "
    "never going to redeem it, eating into margin for no return. Given "
    "that customer acquisition/re-engagement value typically exceeds the "
    "marginal cost of one unredeemed offer, this business slightly favors "
    "RECALL over precision -- better to over-target and accept some wasted "
    "offers than to systematically under-reach likely completers.")


# Task 5b: Threshold sensitivity

log("\n" + "=" * 70)
log("TASK 5b: Threshold sensitivity (0.30-0.70)")
log("=" * 70)
thresholds = np.linspace(0.30, 0.70, 5)
rows = []
for t in thresholds:
    pred_t = (pred_proba >= t).astype(int)
    rows.append({
        "threshold": round(t, 2),
        "precision": precision_score(y_clf_test, pred_t, zero_division=0),
        "recall": recall_score(y_clf_test, pred_t, zero_division=0),
        "f1": f1_score(y_clf_test, pred_t, zero_division=0),
    })
threshold_df = pd.DataFrame(rows)
log(f"\n{threshold_df}")
best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
log(f"\nF1-maximizing threshold: {best_row['threshold']} (F1={best_row['f1']:.3f})")
log(f"Given the recall-favoring business framing above, a threshold at or "
    f"slightly below the F1-optimal point ({best_row['threshold']}) is "
    f"reasonable -- it avoids over-correcting toward precision at the "
    f"cost of missing likely completers.")


# Task 6: Regularization experiment (C=0.01 vs C=1.0)

log("\n" + "=" * 70)
log("TASK 6: Regularization experiment (Logistic Regression C=0.01 vs C=1.0)")
log("=" * 70)
reg_rows = []
for c in [0.01, 1.0]:
    m = LogisticRegression(max_iter=1000, C=c, class_weight=class_weight_arg, random_state=42)
    m.fit(Xc_train_scaled, y_clf_train)
    p = m.predict(Xc_test_scaled)
    pp = m.predict_proba(Xc_test_scaled)[:, 1]
    reg_rows.append({
        "C": c,
        "accuracy": accuracy_score(y_clf_test, p),
        "precision": precision_score(y_clf_test, p, zero_division=0),
        "recall": recall_score(y_clf_test, p, zero_division=0),
        "f1": f1_score(y_clf_test, p, zero_division=0),
        "auc": roc_auc_score(y_clf_test, pp),
        "mean_abs_coef": np.mean(np.abs(m.coef_)),
    })
reg_df = pd.DataFrame(reg_rows)
log(f"\n{reg_df}")
log("C is the inverse of regularization strength in scikit-learn's "
    "LogisticRegression (smaller C = stronger L2 penalty). C=0.01 heavily "
    "shrinks coefficients toward zero (lower mean_abs_coef), producing a "
    "simpler, more conservative decision boundary; C=1.0 allows the model "
    "to fit the training data more closely, at some risk of overfitting "
    "to noise.")


# Task 7: Bootstrap AUC confidence interval

log("\n" + "=" * 70)
log("TASK 7: Bootstrap AUC 95% CI (500 resamples)")
log("=" * 70)
rng = np.random.RandomState(42)
n = len(y_clf_test)
y_test_arr = y_clf_test.values
boot_aucs = []
for i in range(500):
    idx = rng.randint(0, n, n)
    y_b = y_test_arr[idx]
    p_b = pred_proba[idx]
    if len(np.unique(y_b)) < 2:
        continue
    boot_aucs.append(roc_auc_score(y_b, p_b))
boot_aucs = np.array(boot_aucs)
ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
mean_auc_diff = boot_aucs.mean() - 0.5  # difference from random-chance baseline (AUC=0.5)
log(f"Bootstrap resamples used: {len(boot_aucs)}")
log(f"Mean bootstrap AUC: {boot_aucs.mean():.4f}")
log(f"Mean AUC difference from random baseline (0.5): {mean_auc_diff:.4f}")
log(f"95% CI for AUC: [{ci_low:.4f}, {ci_high:.4f}]")
excludes_zero = ci_low > 0.5
log(f"Does the 95% CI exclude 0.5 (random-chance baseline)? {excludes_zero} "
    f"-- {'the model performs significantly better than random guessing.' if excludes_zero else 'the model does NOT significantly outperform random guessing.'}")

with open("part2_output_log.txt", "w") as f:
    f.write("\n".join(log_lines))

print("\nPart 2 complete.")
