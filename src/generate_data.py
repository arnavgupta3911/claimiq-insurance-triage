from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N = 5000
RNG = np.random.default_rng(SEED)

CLAIM_TYPES = ["Medical", "Disability", "Life"]

claim_amount = RNG.lognormal(mean=8.3, sigma=0.7, size=N).clip(200, 50_000).round(2)
claim_type = RNG.choice(CLAIM_TYPES, size=N, p=[0.55, 0.25, 0.20])
policy_age_months = RNG.integers(1, 241, N)

documents_required = RNG.integers(2, 7, N)
documents_missing = RNG.binomial(n=documents_required, p=0.18)
documents_submitted = documents_required - documents_missing

days_since_incident = RNG.integers(0, 181, N)
previous_claims = RNG.poisson(0.8, N).clip(0, 7)

score = (
    -2.5
    + 0.000065 * claim_amount
    + 0.75 * documents_missing
    + 0.008 * days_since_incident
    + 0.20 * previous_claims
    - 0.009 * policy_age_months
    + 0.25 * (claim_type == "Disability")
    + RNG.normal(0, 0.5, N)
)
probability = 1 / (1 + np.exp(-score))
requires_review = RNG.binomial(1, probability)

df = pd.DataFrame({
    "claim_id": [f"CLM-{i:04d}" for i in range(1, N + 1)],
    "claim_amount": claim_amount,
    "claim_type": claim_type,
    "policy_age_months": policy_age_months,
    "documents_submitted": documents_submitted,
    "documents_required": documents_required,
    "days_since_incident": days_since_incident,
    "previous_claims": previous_claims,
    "requires_review": requires_review,
})

if __name__ == "__main__":
    output = Path("data/synthetic_claims.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved {len(df)} claims to {output}")
    print("\nLabel balance:")
    print(df["requires_review"].value_counts(normalize=True).round(3))
    print("\nMissing values:", int(df.isna().sum().sum()))
    print("Invalid document counts:", int((df["documents_submitted"] > df["documents_required"]).sum()))
    print("\nNumeric summary:")
    print(df.describe().round(2).T)
    print("\nReview rate by claim type:")
    print(df.groupby("claim_type")["requires_review"].mean().round(3))
