from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features.parquet"

logger = logging.getLogger("talentmind_app")


@st.cache_data(show_spinner=False, ttl=3600)
def analyze_candidates(job_description: str, top_k: int) -> pd.DataFrame:
    from src.explainability import explain_shortlist
    from src.ranking_model import rank_candidates

    ranked = rank_candidates(job_description=job_description, top_n=top_k)
    feature_frame = pd.read_parquet(FEATURES_PATH)
    explanations = explain_shortlist(ranked_candidates=ranked, feature_frame=feature_frame)

    explanation_frame = pd.DataFrame(explanations)
    merged = ranked.merge(feature_frame, on="candidate_id", how="left")
    merged = merged.merge(
        explanation_frame[["candidate_id", "confidence", "summary", "strengths", "weaknesses", "risk_flags"]],
        on="candidate_id",
        how="left",
    )

    merged = merged.sort_values("final_score", ascending=False).reset_index(drop=True)
    merged.insert(1, "rank", range(1, len(merged) + 1))
    merged = merged[
        [
            "candidate_id",
            "rank",
            "semantic_similarity",
            "final_score",
            "confidence",
            "summary",
            "authenticity_score",
            "production_ai_experience_score",
            "recruiter_interest_score",
            "career_transition_score",
            "strengths",
            "weaknesses",
            "risk_flags",
        ]
    ].copy()
    return merged


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_radar_chart(candidate_row: pd.Series) -> go.Figure:
    labels = ["Semantic Match", "Authenticity", "Production AI", "Recruiter Interest", "Career Transition"]
    values = [
        min(max(_safe_float(candidate_row.get("semantic_similarity", 0.0)) * 100.0, 0.0), 100.0),
        min(max(_safe_float(candidate_row.get("authenticity_score", 0.0)), 0.0), 100.0),
        min(max(_safe_float(candidate_row.get("production_ai_experience_score", 0.0)), 0.0), 100.0),
        min(max(_safe_float(candidate_row.get("recruiter_interest_score", 0.0)), 0.0), 100.0),
        min(max(_safe_float(candidate_row.get("career_transition_score", 0.0)), 0.0), 100.0),
    ]
    values += values[:1]
    labels += labels[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            line_color="#6EE7F9",
            fillcolor="rgba(110,231,249,0.25)",
            marker=dict(color="#6EE7F9"),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        margin=dict(l=30, r=30, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
    )
    return fig


def _render_metrics(row: pd.Series) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Match %", f"{row['semantic_similarity'] * 100:.1f}%")
    with col2:
        st.metric("Confidence %", f"{int(row['confidence'])}%")
    with col3:
        st.metric("Final Score", f"{row['final_score']:.3f}")


def _render_list(items: Any) -> None:
    if isinstance(items, list) and items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.markdown("- No items reported")


st.set_page_config(page_title="TalentMind AI", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg, #060816 0%, #111827 100%); color: #F9FAFB;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stSidebar"] {background: #0F172A;}
    .stTextInput > div > div > input, .stSlider > div > div > div, .stSelectbox > div > div > div {
        background: #111827;
        color: #F9FAFB;
        border: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("TalentMind AI")
st.subheader("Intelligent Candidate Ranking Platform")

with st.sidebar:
    st.header("Candidate Analysis")
    job_description = st.text_area(
        "Paste Job Description",
        height=220,
        placeholder="Describe the role, required skills, and experience level...",
    )
    top_k = st.slider("Top K Candidates", min_value=5, max_value=50, value=10, step=1)
    analyze = st.button("Analyze Candidates", use_container_width=True, type="primary")

    if not job_description.strip():
        st.info("Add a job description to start the ranking workflow.")

if analyze:
    if not job_description.strip():
        st.warning("Please enter a job description before analyzing candidates.")
    else:
        with st.spinner("Analyzing candidates and generating explanations..."):
            submission = analyze_candidates(job_description=job_description, top_k=int(top_k))

        st.success("Analysis complete")

        st.markdown("### Top Ranked Candidates")
        display_df = submission[["rank", "candidate_id", "semantic_similarity", "confidence", "final_score"]].copy()
        display_df["semantic_similarity"] = (display_df["semantic_similarity"] * 100).round(1)
        display_df["confidence"] = display_df["confidence"].astype(int)
        display_df["final_score"] = display_df["final_score"].round(3)
        display_df = display_df.rename(columns={"semantic_similarity": "Match %", "confidence": "Confidence %"})
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download final_submission.csv",
            data=submission[["candidate_id", "rank", "final_score", "confidence", "summary"]].to_csv(index=False).encode("utf-8"),
            file_name="final_submission.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("### Candidate Insights")
        candidate_options = submission["candidate_id"].tolist()
        selected_candidate_id = st.selectbox("Select a candidate", options=candidate_options, index=0)
        selected_row = submission.loc[submission["candidate_id"] == selected_candidate_id].iloc[0]

        _render_metrics(selected_row)

        col_left, col_right = st.columns([1.1, 0.9])
        with col_left:
            st.markdown("#### Radar Chart")
            st.plotly_chart(_build_radar_chart(selected_row), use_container_width=True)
        with col_right:
            st.markdown("#### Why selected")
            st.write(selected_row["summary"])
            st.markdown("#### Strengths")
            _render_list(selected_row.get("strengths"))
            st.markdown("#### Weaknesses")
            _render_list(selected_row.get("weaknesses"))
            st.markdown("#### Risk Flags")
            _render_list(selected_row.get("risk_flags"))
else:
    st.info("Use the sidebar controls to analyze a job description and surface the best matching candidates.")
