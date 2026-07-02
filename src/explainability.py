from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features.parquet"

logger = logging.getLogger("explainability")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_explanation(candidate_row: Dict[str, Any], final_score: float) -> Dict[str, Any]:
    """Create a concise recruiter-friendly explanation for a shortlisted candidate."""
    if not candidate_row:
        raise ValueError("candidate_row cannot be empty")

    candidate_id = str(candidate_row.get("candidate_id", ""))
    semantic_similarity = float(candidate_row.get("semantic_similarity", 0.0))
    authenticity = float(candidate_row.get("authenticity_score", 0.0))
    assessment = float(candidate_row.get("assessment_average", 0.0))
    github_activity = float(candidate_row.get("github_activity_score", 0.0))
    profile_completeness = float(candidate_row.get("profile_completeness_score", 0.0))
    recruiter_interest = float(candidate_row.get("recruiter_interest_score", 0.0))
    recruiter_response = float(candidate_row.get("recruiter_response_rate", 0.0))
    notice_period = float(candidate_row.get("notice_period_days", 0.0)) if "notice_period_days" in candidate_row else 0.0
    production_ai = float(candidate_row.get("production_ai_experience_score", 0.0))
    career_transition = float(candidate_row.get("career_transition_score", 0.0))
    ownership = float(candidate_row.get("ownership_score", 0.0))

    strengths: List[str] = []
    weaknesses: List[str] = []
    risk_flags: List[str] = []

    if production_ai >= 60:
        strengths.append("✓ Strong production AI experience")
    elif production_ai >= 30:
        weaknesses.append("⚠ Limited production experience")

    if semantic_similarity >= 0.6:
        strengths.append("✓ High semantic match")
    elif semantic_similarity < 0.4:
        weaknesses.append("⚠ Weak semantic match")

    if github_activity >= 20:
        strengths.append("✓ Strong GitHub activity")
    elif github_activity < 10:
        weaknesses.append("⚠ Limited GitHub activity")

    if recruiter_interest >= 60:
        strengths.append("✓ Excellent recruiter engagement")
    elif recruiter_interest < 25:
        risk_flags.append("Low recruiter response")

    if career_transition >= 50 or ownership >= 50:
        strengths.append("✓ Proven AI deployment ownership")
    elif career_transition < 25:
        weaknesses.append("⚠ Limited evidence of deployment ownership")

    if authenticity >= 70:
        strengths.append("✓ Strong authenticity signals")
    elif authenticity < 35:
        weaknesses.append("⚠ Skill claims exceed career evidence")

    if assessment < 40:
        weaknesses.append("⚠ Low assessment scores")
    if profile_completeness < 50:
        risk_flags.append("Low profile completeness")
    if notice_period > 45:
        weaknesses.append("⚠ Long notice period")
    if recruiter_response < 0.2:
        risk_flags.append("Low recruiter response")

    confidence_components = [
        semantic_similarity,
        authenticity / 100.0,
        assessment / 100.0,
        github_activity / 100.0,
        profile_completeness / 100.0,
    ]
    confidence_score = round(sum(confidence_components) / len(confidence_components) * 100.0, 1)
    confidence_score = max(0.0, min(100.0, confidence_score))

    if not strengths:
        strengths.append("✓ Solid candidate profile")
    if not weaknesses:
        weaknesses.append("• No major concerns noted")
    if not risk_flags:
        risk_flags.append("• No critical red flags")

    summary = (
        f"Candidate {candidate_id} shows a strong fit for the role with a balanced mix of "
        f"semantic alignment, AI-relevant experience, and recruiter engagement."
    )

    return {
        "candidate_id": candidate_id,
        "confidence": int(round(confidence_score)),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "risk_flags": risk_flags,
        "summary": summary,
    }


def explain_shortlist(
    ranked_candidates: pd.DataFrame,
    feature_frame: Optional[pd.DataFrame] = None,
    output_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Generate recruiter-friendly explanations for each shortlisted candidate."""
    configure_logging()
    if feature_frame is None:
        feature_frame = pd.read_parquet(ROOT_DIR / "data" / "processed" / "features.parquet")

    merged = ranked_candidates.merge(feature_frame, on="candidate_id", how="left")
    explanations: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        candidate_data = row.to_dict()
        explanation = build_explanation(candidate_data, float(row.get("final_score", 0.0)))
        explanations.append(explanation)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(explanations, indent=2), encoding="utf-8")
        logger.info("Saved %s explanations to %s", len(explanations), output_file)

    return explanations


if __name__ == "__main__":
    from ranking_model import rank_candidates

    ranked = rank_candidates(
        "Senior data scientist with strong experience in machine learning, LLMs, deep learning, and production AI systems",
        top_n=10,
    )
    reports = explain_shortlist(ranked)
    print(json.dumps(reports, indent=2))
