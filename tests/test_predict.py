import pandas as pd
import pytest

from src.database import get_claims, init_db, save_claim
from src.predict import PRIORITY, ROUTINE, predict_claim

VALID_CLAIM = {
    "claim_amount": 5200.0,
    "claim_type": "Medical",
    "policy_age_months": 14,
    "documents_submitted": 3,
    "documents_required": 4,
    "days_since_incident": 18,
    "previous_claims": 1,
}


def make_claim(**overrides):
    return {**VALID_CLAIM, **overrides}


def test_dataset_shape_and_document_counts():
    df = pd.read_csv("data/synthetic_claims.csv")
    assert len(df) == 5000
    assert (df["documents_submitted"] <= df["documents_required"]).all()
    assert (df["documents_submitted"] >= 0).all()
    assert df["requires_review"].isin([0, 1]).all()


def test_valid_claim_returns_probability():
    result = predict_claim(VALID_CLAIM)
    assert 0 <= result["review_score"] <= 1
    assert result["review_queue"] in (PRIORITY, ROUTINE)


def test_threshold_controls_queue():
    assert predict_claim(VALID_CLAIM, threshold=0.0)["review_queue"] == PRIORITY
    assert predict_claim(VALID_CLAIM, threshold=1.0)["review_queue"] == ROUTINE


def test_missing_documents_raise_score():
    complete = predict_claim(make_claim(documents_submitted=4))["review_score"]
    incomplete = predict_claim(make_claim(documents_submitted=0))["review_score"]
    assert incomplete > complete


@pytest.mark.parametrize("overrides", [
    {"claim_amount": 0},
    {"claim_amount": -100},
    {"claim_amount": "lots"},
    {"claim_type": "Auto"},
    {"policy_age_months": -1},
    {"days_since_incident": -5},
    {"previous_claims": -1},
    {"documents_submitted": 5, "documents_required": 4},
    {"documents_required": 0, "documents_submitted": 0},
    {"previous_claims": 1.5},
])
def test_invalid_claims_rejected(overrides):
    with pytest.raises(ValueError):
        predict_claim(make_claim(**overrides))


def test_missing_field_rejected():
    claim = dict(VALID_CLAIM)
    del claim["claim_type"]
    with pytest.raises(ValueError):
        predict_claim(claim)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        predict_claim(VALID_CLAIM, threshold=threshold)


def test_save_and_retrieve_claims(tmp_path):
    db = tmp_path / "test.db"
    init_db(db)
    assert get_claims(db).empty

    low = make_claim(documents_submitted=4, days_since_incident=0)
    high = make_claim(documents_submitted=0, days_since_incident=180)
    save_claim(low, predict_claim(low), db)
    save_claim(high, predict_claim(high), db)

    saved = get_claims(db)
    assert len(saved) == 2
    assert saved["review_score"].is_monotonic_decreasing
    assert saved.iloc[0]["documents_submitted"] == 0


def test_sql_injection_is_stored_as_text(tmp_path):
    db = tmp_path / "test.db"
    init_db(db)
    claim = make_claim()
    save_claim(claim, {"review_score": 0.5, "review_queue": "x'); DROP TABLE claims; --"}, db)
    assert len(get_claims(db)) == 1
