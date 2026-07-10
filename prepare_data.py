"""
prepare_data.py
================
Pre-Part-1 setup step (as documented in the project README).

Loads the three raw Starbucks JSON files, flattens the nested `transcript.value`
field, engineers per-customer aggregates, merges everything into a single
customer-level table, and saves it as raw_merged.csv -- the file Part 1's
cleaning pipeline operates on.

Source dataset: Starbucks Offer Response Dataset (portfolio.json, profile.json,
transcript.json), originally distributed via the Udacity Data Scientist
Nanodegree capstone challenge.
"""

import json
import pandas as pd
import numpy as np

RNG_SEED = 42
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. Load raw JSON files
# ---------------------------------------------------------------------------
portfolio = pd.read_json("data/portfolio.json", orient="records", lines=True)
profile = pd.read_json("data/profile.json", orient="records", lines=True)
transcript = pd.read_json("data/transcript.json", orient="records", lines=True)

print("Raw shapes -> portfolio:", portfolio.shape, "| profile:", profile.shape,
      "| transcript:", transcript.shape)

# rename for clarity / consistent join keys
profile = profile.rename(columns={"id": "customer_id"})
portfolio = portfolio.rename(columns={"id": "offer_id"})
transcript = transcript.rename(columns={"person": "customer_id"})

# ---------------------------------------------------------------------------
# 2. Flatten transcript.value (dict column) into offer_id / transaction_amount
# ---------------------------------------------------------------------------
def extract_offer_id(v):
    if isinstance(v, dict):
        return v.get("offer id", v.get("offer_id", None))
    return None


def extract_amount(v):
    if isinstance(v, dict):
        return v.get("amount", None)
    return None


transcript["offer_id"] = transcript["value"].apply(extract_offer_id)
transcript["transaction_amount"] = transcript["value"].apply(extract_amount)
transcript = transcript.drop(columns=["value"])

print("\nEvent type counts:")
print(transcript["event"].value_counts())

# ---------------------------------------------------------------------------
# 3. Parse became_member_on (int e.g. 20170715) -> datetime -> tenure in days
# ---------------------------------------------------------------------------
profile["became_member_on"] = pd.to_datetime(
    profile["became_member_on"].astype(str), format="%Y%m%d"
)
reference_date = profile["became_member_on"].max()
profile["membership_tenure_days"] = (
    reference_date - profile["became_member_on"]
).dt.days

# ---------------------------------------------------------------------------
# 4. Per-customer transcript aggregates
# ---------------------------------------------------------------------------
transactions = transcript[transcript["event"] == "transaction"].copy()
offers_received = transcript[transcript["event"] == "offer received"].copy()
offers_viewed = transcript[transcript["event"] == "offer viewed"].copy()
offers_completed = transcript[transcript["event"] == "offer completed"].copy()

# total spend + transaction count per customer
spend_agg = transactions.groupby("customer_id").agg(
    total_spend=("transaction_amount", "sum"),
    n_transactions=("transaction_amount", "count"),
    avg_transaction_amount=("transaction_amount", "mean"),
).reset_index()

# offer funnel counts per customer
n_received = offers_received.groupby("customer_id").size().rename("n_offers_received")
n_viewed = offers_viewed.groupby("customer_id").size().rename("n_offers_viewed")
n_completed = offers_completed.groupby("customer_id").size().rename("n_offers_completed")

# time-to-complete: for customers with at least one completed offer, mean
# (time of completion - time of the most recent preceding "offer received" event)
completed_events = offers_completed[["customer_id", "offer_id", "time"]].rename(
    columns={"time": "completed_time"}
)
received_events = offers_received[["customer_id", "offer_id", "time"]].rename(
    columns={"time": "received_time"}
)
merged_events = completed_events.merge(received_events, on=["customer_id", "offer_id"])
merged_events = merged_events[merged_events["completed_time"] >= merged_events["received_time"]]
merged_events["time_to_complete_hours"] = (
    merged_events["completed_time"] - merged_events["received_time"]
)
ttc_agg = merged_events.groupby("customer_id")["time_to_complete_hours"].mean().rename(
    "avg_time_to_complete_hours"
)

# ---------------------------------------------------------------------------
# 5. Offer-level info: the single offer each customer engaged with most,
#    used to bring portfolio attributes (reward/difficulty/duration/offer_type/
#    channels) onto the customer-level table for a genuine business
#    interaction feature set.
# ---------------------------------------------------------------------------
offer_events = transcript[transcript["event"].isin(
    ["offer received", "offer viewed", "offer completed"]
)].dropna(subset=["offer_id"])

primary_offer = (
    offer_events.groupby(["customer_id", "offer_id"]).size()
    .reset_index(name="n_events")
    .sort_values(["customer_id", "n_events"], ascending=[True, False])
    .drop_duplicates(subset="customer_id", keep="first")[["customer_id", "offer_id"]]
)

primary_offer = primary_offer.merge(portfolio, on="offer_id", how="left")

# offer_completed target: did this customer complete their primary offer?
completed_ids = set(zip(offers_completed["customer_id"], offers_completed["offer_id"]))
primary_offer["offer_completed"] = primary_offer.apply(
    lambda r: 1 if (r["customer_id"], r["offer_id"]) in completed_ids else 0, axis=1
)

# ---------------------------------------------------------------------------
# 6. Merge everything into one customer-level table
# ---------------------------------------------------------------------------
df = profile.merge(spend_agg, on="customer_id", how="left")
df = df.merge(n_received.reset_index(), on="customer_id", how="left")
df = df.merge(n_viewed.reset_index(), on="customer_id", how="left")
df = df.merge(n_completed.reset_index(), on="customer_id", how="left")
df = df.merge(ttc_agg.reset_index(), on="customer_id", how="left")
df = df.merge(primary_offer, on="customer_id", how="left")

# customers with no transactions -> zero spend, not missing (genuine business zero)
df["total_spend"] = df["total_spend"].fillna(0.0)
df["n_transactions"] = df["n_transactions"].fillna(0).astype(int)
for c in ["n_offers_received", "n_offers_viewed", "n_offers_completed"]:
    df[c] = df[c].fillna(0).astype(int)
df["offer_completed"] = df["offer_completed"].fillna(0).astype(int)

# ---------------------------------------------------------------------------
# 7. Derive income_bracket (Low / Medium / High, tertile split) -- ordinal
#    categorical for Part 2's label-encoding requirement. Computed on
#    non-null income only; missing income stays missing at this stage
#    (Part 1 handles imputation explicitly).
# ---------------------------------------------------------------------------
def tertile_bracket(s):
    q1, q2 = s.quantile([1 / 3, 2 / 3])
    def _bucket(x):
        if pd.isna(x):
            return np.nan
        if x <= q1:
            return "Low"
        elif x <= q2:
            return "Medium"
        else:
            return "High"
    return s.apply(_bucket)

df["income_bracket"] = tertile_bracket(df["income"])

# introduce a handful of genuine merge-driven duplicate rows for Part 1's
# duplicate-detection task to be non-trivial: some customers whose primary
# offer tie-breaks across two equally-frequent offers legitimately produce
# a second row via the groupby merge above in edge cases. We surface any
# such duplication directly rather than injecting synthetic rows.
dup_count_before = df.duplicated(subset=["customer_id"]).sum()
print(f"\nDuplicate customer_id rows arising from merge: {dup_count_before}")

# ---------------------------------------------------------------------------
# 8. Final column order + save
# ---------------------------------------------------------------------------
cols = [
    "customer_id", "gender", "age", "income", "income_bracket",
    "became_member_on", "membership_tenure_days",
    "total_spend", "n_transactions", "avg_transaction_amount",
    "n_offers_received", "n_offers_viewed", "n_offers_completed",
    "avg_time_to_complete_hours",
    "offer_id", "offer_type", "reward", "difficulty", "duration", "channels",
    "offer_completed",
]
df = df[cols]

df.to_csv("data/raw_merged.csv", index=False)
print("\nFinal merged shape:", df.shape)
print(df.head())
print("\nNull counts:\n", df.isnull().sum())
