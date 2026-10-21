import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.database import get_claims, init_db, save_claim
from src.predict import CLAIM_TYPES, DEFAULT_THRESHOLD, FEATURES, PRIORITY, ROUTINE, predict_claim

METRICS_PATH = Path(__file__).resolve().parent / "models" / "metrics.json"
DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_claims.csv"

DISCLAIMER = (
    "**Synthetic demonstration only.** Model scores reflect patterns in artificially generated data. "
    "They are not calibrated estimates of real insurance risk and must not be used for actual claims decisions."
)

st.set_page_config(page_title="ClaimIQ", page_icon="📋", layout="wide")
init_db()


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


def seed_sample_claims(n, threshold):
    sample = pd.read_csv(DATA_PATH).sample(n)
    for record in sample[FEATURES].to_dict(orient="records"):
        claim = {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}
        save_claim(claim, predict_claim(claim, threshold))


with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Review threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05,
        help="Claims scoring at or above this are sent to the priority queue. "
             "0.5 is a demonstration default, not an operational policy.",
    )
    st.caption("Lower thresholds catch more review cases (higher recall) at the cost of more false alarms.")
    st.divider()
    if st.button("Add 25 sample claims", help="Score 25 random rows from the synthetic dataset and save them."):
        seed_sample_claims(25, threshold)
        st.success("Added 25 sample claims.")

st.title("ClaimIQ")
st.caption("AI-assisted insurance claims triage · prototype")
st.warning(DISCLAIMER)

submit_tab, queue_tab, analytics_tab = st.tabs(["Submit Claim", "Review Queue", "Analytics"])

with submit_tab:
    form_col, result_col = st.columns([3, 2], gap="large")

    with form_col:
        with st.form("claim_form"):
            st.subheader("New claim")
            c1, c2 = st.columns(2)
            claim_type = c1.selectbox("Claim type", CLAIM_TYPES)
            claim_amount = c2.number_input("Claim amount ($)", min_value=0.0, value=5200.0, step=100.0)
            policy_age = c1.number_input("Policy age (months)", min_value=0, value=14, step=1)
            days_since = c2.number_input("Days since incident", min_value=0, value=18, step=1)
            docs_required = c1.number_input("Documents required", min_value=1, value=4, step=1)
            docs_submitted = c2.number_input("Documents submitted", min_value=0, value=3, step=1)
            previous = c1.number_input("Previous claims", min_value=0, value=1, step=1)
            save = st.checkbox("Save to review queue", value=True)
            submitted = st.form_submit_button("Score claim", type="primary")

    with result_col:
        if submitted:
            claim = {
                "claim_amount": float(claim_amount),
                "claim_type": claim_type,
                "policy_age_months": int(policy_age),
                "documents_submitted": int(docs_submitted),
                "documents_required": int(docs_required),
                "days_since_incident": int(days_since),
                "previous_claims": int(previous),
            }
            try:
                result = predict_claim(claim, threshold)
            except ValueError as e:
                st.error(f"Invalid claim: {e}")
            else:
                st.subheader("Result")
                if result["review_queue"] == PRIORITY:
                    st.error(f"**{PRIORITY}**")
                else:
                    st.success(f"**{ROUTINE}**")
                st.metric("Simulated model score", f"{result['review_score']:.2f}")
                st.progress(result["review_score"])
                st.caption(f"Threshold: {threshold:.2f}")

                missing = claim["documents_required"] - claim["documents_submitted"]
                notes = []
                if missing:
                    notes.append(f"{missing} supporting document(s) missing")
                if claim["days_since_incident"] > 90:
                    notes.append("submitted more than 90 days after the incident")
                if claim["policy_age_months"] < 12:
                    notes.append("policy is less than a year old")
                if notes:
                    st.markdown("**Things a reviewer may want to check:**\n" + "\n".join(f"- {n}" for n in notes))

                if save:
                    row_id = save_claim(claim, result)
                    st.info(f"Saved as CLM-{row_id:04d}.")
        else:
            st.info("Fill in the form and press **Score claim**.")

    st.divider()
    st.subheader("What drives the model's scores?")
    metrics = load_metrics()
    coefs = pd.DataFrame(metrics["coefficients"]).set_index("feature")["coefficient"].sort_values()
    st.bar_chart(coefs, horizontal=True, sort="-coefficient", x_label="", y_label="Standardised coefficient")
    st.caption(
        "Global view of the fitted logistic regression. Positive coefficients are associated with higher review "
        "scores, negative with lower, holding other inputs fixed. These describe associations the model learned "
        "from synthetic data: they are not causal, and they don't fully explain any individual prediction. "
        "documents_submitted and documents_required pull in opposite directions because together they encode "
        "the number of missing documents."
    )

with queue_tab:
    claims = get_claims()
    st.subheader("Review queue")
    if claims.empty:
        st.info("No saved claims yet. Submit one, or use **Add 25 sample claims** in the sidebar.")
    else:
        claims.insert(0, "claim_id", claims["id"].map(lambda i: f"CLM-{i:04d}"))
        f1, f2, f3 = st.columns([2, 2, 2])
        search = f1.text_input("Search claim ID", placeholder="e.g. CLM-0007")
        types = f2.multiselect("Claim type", CLAIM_TYPES, default=list(CLAIM_TYPES))
        queues = f3.multiselect("Queue", [PRIORITY, ROUTINE], default=[PRIORITY, ROUTINE])

        view = claims[claims["claim_type"].isin(types) & claims["review_queue"].isin(queues)]
        if search:
            view = view[view["claim_id"].str.contains(search.strip(), case=False)]

        st.caption(f"Showing {len(view)} of {len(claims)} claims, highest score first. Click a column header to re-sort.")
        st.dataframe(
            view.drop(columns="id"),
            hide_index=True,
            use_container_width=True,
            column_config={
                "claim_id": "Claim ID",
                "claim_type": "Type",
                "claim_amount": st.column_config.NumberColumn("Amount", format="$%.0f"),
                "policy_age_months": "Policy age (mo)",
                "documents_submitted": "Docs submitted",
                "documents_required": "Docs required",
                "days_since_incident": "Days since incident",
                "previous_claims": "Prev. claims",
                "review_score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.2f"),
                "review_queue": "Recommendation",
                "created_at": "Saved at (UTC)",
            },
        )
        st.caption("Queue labels use the threshold in effect when each claim was saved.")

with analytics_tab:
    claims = get_claims()
    st.subheader("Saved claims")
    if claims.empty:
        st.info("Analytics will appear once claims are saved.")
    else:
        flagged = int((claims["review_queue"] == PRIORITY).sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total claims", len(claims))
        m2.metric("Review flagged", flagged)
        m3.metric("Routine queue", len(claims) - flagged)
        m4.metric("% flagged", f"{flagged / len(claims):.0%}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Claims by type and queue**")
            by_type = claims.groupby(["claim_type", "review_queue"]).size().unstack(fill_value=0)
            st.bar_chart(by_type, x_label="Claim type", y_label="Claims")
        with c2:
            st.markdown("**Review score distribution**")
            bins = pd.cut(claims["review_score"], bins=[i / 10 for i in range(11)], include_lowest=True)
            hist = bins.value_counts().sort_index()
            hist.index = [f"{max(iv.left, 0):.1f}–{iv.right:.1f}" for iv in hist.index]
            st.bar_chart(hist, x_label="Score", y_label="Claims")

    st.divider()
    st.subheader("Model evaluation (held-out test set, 1,000 synthetic claims)")
    metrics = load_metrics()
    comparison = pd.DataFrame({"Logistic regression": metrics["model"], "Majority-class baseline": metrics["baseline"]})
    st.dataframe(comparison.round(3), use_container_width=True)
    st.markdown("**Precision / recall at different thresholds**")
    st.dataframe(pd.DataFrame(metrics["thresholds"]).round(3), hide_index=True, use_container_width=True)
