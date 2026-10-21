from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "claim_model.joblib"

FEATURES = [
    "claim_amount",
    "claim_type",
    "policy_age_months",
    "documents_submitted",
    "documents_required",
    "days_since_incident",
    "previous_claims",
]
CLAIM_TYPES = ("Medical", "Disability", "Life")
INTEGER_FIELDS = [f for f in FEATURES if f not in ("claim_amount", "claim_type")]
DEFAULT_THRESHOLD = 0.5

PRIORITY = "Prioritise review"
ROUTINE = "Routine queue"


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def validate_claim(claim):
    missing = [f for f in FEATURES if f not in claim]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    if claim["claim_type"] not in CLAIM_TYPES:
        raise ValueError("Invalid claim type")

    amount = claim["claim_amount"]
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise ValueError("Claim amount must be a number")
    if amount <= 0:
        raise ValueError("Claim amount must be positive")

    for field in INTEGER_FIELDS:
        value = claim[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be a whole number")
        if value < 0:
            raise ValueError(f"{field} cannot be negative")

    if claim["documents_required"] < 1:
        raise ValueError("At least one document must be required")
    if claim["documents_submitted"] > claim["documents_required"]:
        raise ValueError("Documents submitted cannot exceed documents required")


def predict_claim(claim, threshold=DEFAULT_THRESHOLD):
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    validate_claim(claim)

    row = pd.DataFrame([{f: claim[f] for f in FEATURES}])
    score = float(load_model().predict_proba(row)[0, 1])
    return {
        "review_score": score,
        "review_queue": PRIORITY if score >= threshold else ROUTINE,
    }


if __name__ == "__main__":
    example = {
        "claim_amount": 5200,
        "claim_type": "Medical",
        "policy_age_months": 14,
        "documents_submitted": 3,
        "documents_required": 4,
        "days_since_incident": 18,
        "previous_claims": 1,
    }
    print(predict_claim(example))
