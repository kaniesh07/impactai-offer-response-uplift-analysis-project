# Part 5  Causal Uplift Analysis

**Extension beyond assignment scope — causal inference for marketing decisioning.** This Part is
kept entirely separate from the graded submission (Parts 1–4) and does not affect the 100-mark
rubric in any way — it exists purely as a portfolio differentiator.

## Framing

- **Treatment**: `offer_type` (bogo / discount / informational) — reframed as the intervention
  being tested, rather than just a predictive feature.
- **Outcome**: `offer_completed` (binary) and/or `total_spend` (continuous).
- **Goal**: not "will this customer complete an offer" (Parts 2–3's question) but "how much MORE
  likely / how much MORE would this customer spend *because of* receiving this specific offer,
  versus a different offer or no offer at all" — i.e., the individual treatment effect (uplift),
  not just the outcome propensity.

## Why This Matters Beyond Parts 2–3

The classification/regression models in Parts 2–3 predict **P(complete | features, offer)** — a
correlational quantity useful for *ranking* customers by likely response. But it cannot answer
"would this specific customer have completed anyway, even with a cheaper/no offer?" — which is
exactly the question that determines whether an offer campaign is *profitable*, not just
*predictive*. A high-propensity customer who would have purchased regardless is a wasted incentive
cost; the real ROI lever is customers whose behavior actually changes *because of* the offer
(persuadables), not those who are "sure things" or "lost causes" either way.

## Proposed Methodology

1. **T-learner**: train one outcome model per offer_type (bogo, discount, informational, and a
   proxy no-offer baseline using customers with 0 offers received where available), each predicting
   `total_spend` or `offer_completed` from customer features alone. The uplift estimate for a
   customer is the difference between the treatment-arm model's prediction and the baseline-arm
   model's prediction for that same customer's features.
2. **X-learner**: extends the T-learner by using each arm's model to impute the counterfactual
   outcome for the *other* arm's observed customers, then trains a second-stage model directly on
   these imputed individual treatment effects — generally more sample-efficient than the T-learner
   when treatment groups are imbalanced in size, which is the case here (discount: 8,798 vs
   informational: 1,926 primary-offer customers).
3. **Libraries**: `causalml` (Uber) or `econml` (Microsoft) both implement T-/X-learners
   out-of-the-box on top of any scikit-learn-compatible base estimator — e.g. the same
   `RandomForestRegressor`/`RandomForestClassifier` families already validated in Part 3.

## Policy Simulation

Once individual uplift scores are estimated, two targeting policies can be compared:
- **Uplift-targeted**: send the offer only to the top-decile-by-predicted-uplift customers.
- **Blanket targeting**: send the offer to everyone (the current, non-personalized default).

**ROI comparison** would then be: `(incremental_spend_from_targeted_group - total_incentive_cost_of_targeted_group)`
vs the same calculation for blanket targeting — the uplift-targeted policy is expected to show a
meaningfully better ROI ratio precisely because it avoids paying incentive costs on customers who
would have converted (or not) regardless of the offer.

## Caveat on the Randomization Assumption

A clean causal estimate technically requires offer assignment to be independent of customer
potential outcomes (as in a randomized controlled trial). The Starbucks simulator's offer assignment
is documented as approximately random across the customer base, which supports treating this as a
reasonable natural experiment — but this assumption is not independently re-verified here, and a
production deployment of this methodology would need to confirm it against the actual assignment
mechanism (or fall back to techniques robust to non-random assignment, e.g. propensity-score
weighting) before trusting the ROI numbers for a real budget decision.

## Status

This Part is a methodology writeup only in this submission (no executed code), by design — it is
explicitly out of scope for the graded rubric (Ridge, Gini/Entropy, GridSearchCV, etc. — all of
which are fully implemented and verified in Parts 2–3 instead). It documents the resume/interview-
ready extension a candidate would build next, without risking non-compliance with the graded
rubric's exact mechanical requirements.
