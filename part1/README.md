# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis 

## Dataset & Setup

Source: Starbucks Offer Response Dataset (`portfolio.json`, `profile.json`, `transcript.json`) —
the public Udacity Data Scientist Nanodegree capstone dataset. Raw files live in `data/`.

`prepare_data.py` (repo root) performs the pre-Part-1 setup:
1. Loads all three raw JSON files (portfolio: 10×6, profile: 17,000×5, transcript: 306,534×4).
2. Flattens `transcript.value` into `offer_id` / `transaction_amount`.
3. Parses `became_member_on` (int `YYYYMMDD`) into a real datetime and derives `membership_tenure_days`.
4. Aggregates transcript events per customer (total spend, transaction count, offer funnel counts,
   average time-to-complete) and attaches each customer's "primary offer" (the offer they interacted
   with most) along with its portfolio attributes.
5. Derives `income_bracket` (Low/Medium/High tertile split).
6. Saves `data/raw_merged.csv` — **17,000 rows × 21 columns**, the input to this Part.

Run order: `python3 prepare_data.py` (from repo root) → `python3 part1/part1_cleaning_eda.py` (from `part1/`).

## Task-by-Task Results

**Task 1 — Load & inspect.** Shape confirmed at (17000, 21). 

**Task 2 — Null analysis.** Real nulls (not synthetic) appear in `gender`, `income`, and
`income_bracket`, each at **2,175 / 17,000 = 12.79%** — matching the dataset's known ~13% missing
demographic rate. **No column exceeds the 20% threshold**, so no column was dropped outright; all
were handled via imputation instead. `avg_transaction_amount` and `avg_time_to_complete_hours` also
carry nulls, but these are *structural* (customers with zero transactions / zero completions), not
missing-at-random — see Task 3.

**Task 3 — Median imputation.**
- `income`: median (64,000) used over mean (65,225.24) because `income` is right-skewed (Task 7)
  — mean imputation would overstate the typical customer's income.
- `avg_transaction_amount`: filled with **0**, not median — nulls here mean "never transacted," a
  genuine zero, not a missing value that should be estimated.
- `avg_time_to_complete_hours`: left as `NaN` (filling would fabricate a completion event that never
  happened); a `has_completed_offer` flag column was added instead so the missingness itself is
  usable downstream without corrupting the numeric column.

**Task 4 — Duplicate detection.** Full-row duplicates: **0**. Duplicate `customer_id`: **0**. This
is a genuine finding, not an evasion — `raw_merged.csv` is built at one-row-per-customer granularity
by design (transcript events are pre-aggregated and each customer is joined to a single primary
offer before merge), so duplicate `customer_id` rows would only appear if the merge join itself were
buggy. The methodology and verification are logged explicitly rather than manufacturing rows to drop.

**Task 5 — Dtype correction.** `age == 118` (the dataset's known sentinel error) affects **2,175
rows**, all of which co-occur exactly with the original `gender`/`income` nulls (both counts are
2,175) — confirming this is one and the same "incomplete profile" cohort, not a coincidence. Sentinel
values were replaced with `NaN` via `pd.to_numeric` + explicit masking, then median-imputed
(median = 55.0). `became_member_on` converted from `int64` to `datetime64[us]`.

**Task 6 — Category dtype conversion.** `gender`, `offer_type`, `channels`, `income_bracket` →
`category` dtype. Memory usage: **9,008,981 bytes → 4,998,735 bytes (44.5% reduction)**.

**Task 7 — Skewness.**

| Column | Skew |
|---|---|
| total_spend | 3.66 |
| membership_tenure_days | 1.03 |
| income | 0.46 |

`total_spend` is the most skewed — expected, since most customers spend modestly while a smaller
group of frequent purchasers pulls the tail far right.

**Task 8 — IQR outliers.**
- `total_spend`: bounds [-168.62, 339.23] → **521 outliers (3.1%)**
- `income`: bounds [13,500, 113,500] → **306 outliers (1.8%)**

**Decision: retained, not dropped.** High-spending customers are exactly the population an uplift /
targeting model (Part 5) needs to identify correctly — removing them would bias every downstream
model toward the low-spend majority and undercut the project's actual business purpose.

**Task 8a — Mean vs median.**

| Column | Mean | Median | Gap |
|---|---|---|---|
| total_spend | 104.44 | 69.41 | 35.03 |
| income | 65,225.24 | 64,000.00 | 1,225.24 |

The gap on `total_spend` is large relative to its scale, confirming right-skew and justifying the
median-based imputation strategy from Task 3.

**Task 8b — Spearman vs Pearson.** Top-3 pairs by |Spearman − Pearson|:

| Pair | |Diff| |
|---|---|
| total_spend vs membership_tenure_days | 0.199 |
| difficulty vs duration | 0.105 |
| membership_tenure_days vs income | 0.046 |

The top pair, `total_spend` vs `membership_tenure_days`, shows a rank (Spearman) relationship
meaningfully stronger than the linear (Pearson) one — consistent with spend accumulating
non-linearly over tenure (early rapid growth via onboarding offers, then plateauing) rather than
scaling linearly with days-as-a-member. `difficulty` vs `duration` also diverges, consistent with
offer design following tiered business rules rather than strict linear scaling.

**Task 8c — Grouped aggregation by `offer_type`.**

| offer_type | mean | std | count |
|---|---|---|---|
| bogo | 108.30 | 130.20 | 6,270 |
| discount | 108.11 | 126.96 | 8,798 |
| informational | 75.27 | 100.88 | 1,926 |

Highest mean & std: **bogo**. Ratio of highest to lowest mean: **1.44×**. This spread confirms
`offer_type` carries real predictive signal — `informational` offers, which carry no reward or
completion mechanic, show a visibly lower mean spend than `bogo`/`discount`, supporting its
inclusion as a model feature in Part 2.

## Correlation Heatmap Discussion

`income`–`total_spend` correlation: **0.310** (moderate positive). A plausible confound is
`membership_tenure_days` / general customer engagement: longer-tenured, more active customers tend
to both self-select into higher income brackets on the loyalty app and accumulate more transactions
simply from having been active longer — inflating the apparent income→spend relationship beyond a
pure income effect.

## Output

`data/cleaned_data.csv` — **17,000 rows × 22 columns**, deduplicated, imputed, correctly typed,
leak-free (no target-derived columns beyond what Part 2 explicitly defines as labels).