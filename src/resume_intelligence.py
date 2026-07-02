from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger("resume_intelligence")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def compute_skill_inflation_score(row: Dict[str, Any]) -> float:
    """Estimate how inflated a candidate's claimed AI skill set appears relative to supporting evidence."""
    claimed_skills = {
        "llm": _safe_float(row.get("llm_experience_score"), 0.0),
        "rag": _safe_float(row.get("rag_vector_database_score"), 0.0),
        "fine_tuning": _safe_float(row.get("deep_learning_score"), 0.0),
        "pytorch": _safe_float(row.get("mlops_score"), 0.0),
    }

    evidence = {
        "production_ai": _safe_float(row.get("production_ai_experience_score"), 0.0),
        "career_transition": _safe_float(row.get("career_transition_score"), 0.0),
        "ownership": _safe_float(row.get("ownership_score"), 0.0),
        "authenticity": _safe_float(row.get("authenticity_score"), 0.0),
        "assessment": _safe_float(row.get("assessment_alignment"), 0.0),
        "github": _safe_float(row.get("github_activity_factor"), 0.0),
    }

    inflation_points = 0.0

    advanced_claims = sum(
        1
        for skill_name, value in claimed_skills.items()
        if skill_name in {"llm", "rag", "fine_tuning", "pytorch"} and value >= 70
    )
    support_signal = (
        evidence["production_ai"] * 0.45
        + evidence["career_transition"] * 0.25
        + evidence["ownership"] * 0.15
        + evidence["authenticity"] * 0.15
    )

    if advanced_claims >= 2 and support_signal < 45:
        inflation_points += 25
    if advanced_claims >= 3 and support_signal < 55:
        inflation_points += 15

    if evidence["assessment"] < 35 and advanced_claims >= 2:
        inflation_points += 15

    if evidence["github"] < 15 and advanced_claims >= 3:
        inflation_points += 15

    if evidence["authenticity"] < 40 and advanced_claims >= 2:
        inflation_points += 10

    if evidence["production_ai"] < 25 and advanced_claims >= 2:
        inflation_points += 10

    score = min(100.0, max(0.0, inflation_points))
    return round(score, 2)


def compute_resume_credibility_score(row: Dict[str, Any]) -> float:
    skill_inflation = _safe_float(row.get("skill_inflation_score"), 0.0)
    evidence_coverage = _safe_float(row.get("evidence_coverage_pct"), 0.0)
    authenticity = _safe_float(row.get("authenticity_score"), 0.0)
    assessment = _safe_float(row.get("assessment_alignment"), 0.0)
    github = _safe_float(row.get("github_activity_factor"), 0.0)

    credibility = 0.4 * (100.0 - skill_inflation) + 0.25 * evidence_coverage + 0.2 * authenticity + 0.15 * ((assessment + github) / 2.0)
    return round(min(100.0, max(0.0, credibility)), 2)


def compute_evidence_coverage_pct(row: Dict[str, Any]) -> float:
    evidence_sources = [
        _safe_float(row.get("production_ai_experience_score"), 0.0) > 0,
        _safe_float(row.get("career_transition_score"), 0.0) > 0,
        _safe_float(row.get("ownership_score"), 0.0) > 0,
        _safe_float(row.get("assessment_alignment"), 0.0) > 0,
        _safe_float(row.get("github_activity_factor"), 0.0) > 0,
    ]
    coverage = sum(evidence_sources) / len(evidence_sources) * 100.0 if evidence_sources else 0.0
    return round(coverage, 2)


def add_resume_intelligence_scores(frame: pd.DataFrame) -> pd.DataFrame:
    configure_logging()
    if frame.empty:
        return frame.copy()

    scored = frame.copy()
    scored["skill_inflation_score"] = scored.apply(lambda row: compute_skill_inflation_score(row.to_dict()), axis=1)

    scored["evidence_coverage_pct"] = scored.apply(lambda row: compute_evidence_coverage_pct(row.to_dict()), axis=1)
    scored["resume_inflation_score"] = scored["skill_inflation_score"].astype(float)
    scored["resume_credibility_score"] = scored.apply(lambda row: compute_resume_credibility_score(row.to_dict()), axis=1)

    scored["resume_inflation_score"] = scored["resume_inflation_score"].clip(lower=0.0, upper=100.0)
    scored["resume_credibility_score"] = scored["resume_credibility_score"].clip(lower=0.0, upper=100.0)
    scored["evidence_coverage_pct"] = scored["evidence_coverage_pct"].clip(lower=0.0, upper=100.0)

    logger.info("Computed resume intelligence scores for %s candidates", len(scored))
    return scored


if __name__ == "__main__":
    sample = pd.DataFrame(
        [
            {
                "candidate_id": "CAND_SAMPLE",
                "llm_experience_score": 85,
                "rag_vector_database_score": 80,
                "deep_learning_score": 75,
                "mlops_score": 65,
                "production_ai_experience_score": 20,
                "career_transition_score": 18,
                "ownership_score": 12,
                "authenticity_score": 30,
                "skill_evidence_alignment": 15,
                "assessment_alignment": 30,
                "github_activity_factor": 10,
                "assessment_average": 35,
                "github_activity_score": 8,
                "keyword_inflation_score": 25,
            }
        ]
    )
    enriched = add_resume_intelligence_scores(sample)
    print(enriched[["candidate_id", "skill_inflation_score", "resume_inflation_score", "resume_credibility_score", "evidence_coverage_pct"]].to_string(index=False))
