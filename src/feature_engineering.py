from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "data" / "raw" / "extracted" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "features.parquet"

logger = logging.getLogger("feature_engineering")

AI_TERMS = {
    "llm": [
        "llm",
        "large language model",
        "generative ai",
        "gpt",
        "chatgpt",
        "claude",
        "anthropic",
        "openai",
        "prompt engineering",
        "prompting",
        "transformer",
        "langchain",
        "llama",
        "huggingface",
    ],
    "ml": [
        "machine learning",
        "ml",
        "statistical modeling",
        "classification",
        "regression",
        "forecasting",
        "recommendation",
        "recommender",
        "experimental design",
        "model evaluation",
    ],
    "deep_learning": [
        "deep learning",
        "neural network",
        "cnn",
        "rnn",
        "transformer",
        "pytorch",
        "tensorflow",
        "keras",
        "computer vision",
        "nlp",
        "natural language processing",
    ],
    "rag_vector_database": [
        "rag",
        "retrieval augmented generation",
        "vector database",
        "faiss",
        "pinecone",
        "weaviate",
        "milvus",
        "chroma",
        "embedding",
        "semantic search",
        "vector search",
    ],
    "mlops": [
        "mlops",
        "model deployment",
        "model monitoring",
        "model serving",
        "model registry",
        "experiment tracking",
        "pipeline orchestration",
        "kubeflow",
        "mlflow",
        "airflow",
        "databricks",
        "ci/cd",
        "continuous integration",
    ],
}

EVIDENCE_WORDS = ["built", "designed", "deployed", "architected", "owned", "led", "optimized"]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _normalize_text(text: Any) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9+#.\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def _count_matches(text: str, phrases: List[str]) -> int:
    if not text:
        return 0
    normalized = _normalize_text(text)
    count = 0
    for phrase in phrases:
        if phrase in normalized:
            count += 1
    return count


def _weighted_score(text: str, phrases: List[str], weights: Optional[Dict[str, float]] = None) -> float:
    if not text:
        return 0.0
    normalized = _normalize_text(text)
    total = 0.0
    for phrase in phrases:
        if phrase in normalized:
            total += weights.get(phrase, 1.0) if weights else 1.0
    return min(100.0, total * 8.0)


def _extract_skill_text(candidate: Dict[str, Any]) -> str:
    skills = candidate.get("skills") or []
    fragments: List[str] = []
    for skill in skills:
        name = skill.get("name") or ""
        proficiency = skill.get("proficiency") or ""
        fragments.append(f"{name} {proficiency}".strip())
    return " ".join(fragment for fragment in fragments if fragment)


def _extract_career_text(candidate: Dict[str, Any]) -> List[str]:
    career_history = candidate.get("career_history") or []
    texts: List[str] = []
    for item in career_history:
        title = item.get("title") or ""
        description = item.get("description") or ""
        texts.append(f"{title} {description}".strip())
    return [text for text in texts if text]


def _extract_semantic_text(candidate: Dict[str, Any]) -> str:
    profile = candidate.get("profile") or {}
    headline = profile.get("headline") or ""
    summary = profile.get("summary") or ""
    skill_text = _extract_skill_text(candidate)
    career_titles = " ".join(item.get("title") or "" for item in candidate.get("career_history") or [] if item.get("title"))
    career_descriptions = " ".join(item.get("description") or "" for item in candidate.get("career_history") or [] if item.get("description"))
    parts = [headline, summary, skill_text, career_titles, career_descriptions]
    return " ".join(part for part in parts if part).strip()


def _score_ai_capability(candidate: Dict[str, Any], capability: str) -> float:
    terms = AI_TERMS[capability]
    skill_text = _extract_skill_text(candidate)
    career_text = " ".join(_extract_career_text(candidate))

    skill_signal = _weighted_score(skill_text, terms)
    career_signal = _weighted_score(career_text, terms)
    combined = (skill_signal * 0.55) + (career_signal * 0.45)

    if capability in {"llm", "deep_learning", "rag_vector_database"}:
        combined += 8.0 if _count_matches(skill_text, terms) and _count_matches(career_text, terms) else 0.0
    if capability == "mlops":
        combined += 10.0 if _count_matches(skill_text, terms) and _count_matches(career_text, terms) else 0.0

    return round(min(100.0, combined), 2)


def _score_career_evidence(candidate: Dict[str, Any]) -> Dict[str, float]:
    career_text = " ".join(_extract_career_text(candidate))
    normalized_career = _normalize_text(career_text)
    evidence_count = sum(1 for word in EVIDENCE_WORDS if word in normalized_career)

    ai_signal = max(
        _score_ai_capability(candidate, "llm"),
        _score_ai_capability(candidate, "ml"),
        _score_ai_capability(candidate, "deep_learning"),
        _score_ai_capability(candidate, "rag_vector_database"),
    )

    production_ai_experience_score = round(min(100.0, (ai_signal * 0.45) + (evidence_count * 10.0)), 2)
    career_transition_score = round(min(100.0, (ai_signal * 0.35) + (evidence_count * 7.0) + (20.0 if ai_signal > 10 else 0.0)), 2)
    ownership_score = round(min(100.0, (evidence_count * 12.0) + (10.0 if "owned" in normalized_career or "led" in normalized_career else 0.0)), 2)

    return {
        "production_ai_experience_score": production_ai_experience_score,
        "career_transition_score": career_transition_score,
        "ownership_score": ownership_score,
    }


def _score_authenticity(candidate: Dict[str, Any], ai_capabilities: Dict[str, float]) -> Dict[str, float]:
    profile = candidate.get("profile") or {}
    headline = _normalize_text(profile.get("headline") or "")
    summary = _normalize_text(profile.get("summary") or "")
    skill_text = _normalize_text(_extract_skill_text(candidate))
    career_text = _normalize_text(" ".join(_extract_career_text(candidate)))

    profile_terms = []
    for capability_name, terms in AI_TERMS.items():
        if _count_matches(headline + " " + summary + " " + skill_text, terms) > 0:
            profile_terms.extend(terms)

    evidence_terms = []
    for capability_name, terms in AI_TERMS.items():
        if _count_matches(career_text, terms) > 0:
            evidence_terms.extend(terms)

    inflation_penalty = max(0, len(set(profile_terms)) - len(set(evidence_terms))) * 6
    keyword_inflation_score = round(max(0.0, min(100.0, 100.0 - inflation_penalty)), 2)

    skill_names = [(_normalize_text(skill.get("name") or "")) for skill in candidate.get("skills") or [] if skill.get("name")]
    skill_name_overlap = 0
    if skill_names:
        for skill_name in skill_names:
            if skill_name and skill_name in career_text:
                skill_name_overlap += 1
        skill_evidence_alignment = round(min(100.0, (skill_name_overlap / max(1, len(skill_names))) * 100.0), 2)
    else:
        skill_evidence_alignment = 0.0

    signals = candidate.get("redrob_signals") or {}
    assessment_scores = signals.get("skill_assessment_scores") or {}
    assessment_values = list(assessment_scores.values()) if isinstance(assessment_scores, dict) else []
    assessment_average = round(sum(_safe_float(v) for v in assessment_values) / max(1, len(assessment_values)), 2) if assessment_values else 0.0
    assessment_alignment = round(min(100.0, (assessment_average * 0.6) + (sum(ai_capabilities.values()) / 5.0 * 0.4)), 2)

    github_activity_value = max(0.0, min(100.0, _safe_float(signals.get("github_activity_score"), 0.0)))
    github_activity_factor = round(github_activity_value, 2)

    authenticity_score = round(
        (keyword_inflation_score * 0.3)
        + (skill_evidence_alignment * 0.3)
        + (assessment_alignment * 0.2)
        + (github_activity_factor * 0.2),
        2,
    )

    return {
        "keyword_inflation_score": keyword_inflation_score,
        "skill_evidence_alignment": skill_evidence_alignment,
        "assessment_alignment": assessment_alignment,
        "github_activity_factor": github_activity_factor,
        "authenticity_score": authenticity_score,
    }


def _score_behavioral_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    signals = candidate.get("redrob_signals") or {}
    profile = candidate.get("profile") or {}

    profile_completeness_score = signals.get("profile_completeness_score")
    if profile_completeness_score is None:
        required_fields = [
            profile.get("headline"),
            profile.get("summary"),
            profile.get("current_title"),
            profile.get("current_company"),
            profile.get("location"),
            profile.get("country"),
            profile.get("years_of_experience"),
            (candidate.get("career_history") or []),
            (candidate.get("education") or []),
            (candidate.get("skills") or []),
        ]
        completeness_ratio = sum(1 for field in required_fields if field) / len(required_fields)
        profile_completeness_score = round(completeness_ratio * 100.0, 2)
    else:
        profile_completeness_score = round(_safe_float(profile_completeness_score, 0.0), 2)

    recruiter_response_rate = round(_safe_float(signals.get("recruiter_response_rate"), 0.0) * 100.0, 2)
    github_activity_score = round(max(0.0, min(100.0, _safe_float(signals.get("github_activity_score"), 0.0))), 2)

    assessment_scores = signals.get("skill_assessment_scores") or {}
    assessment_values = list(assessment_scores.values()) if isinstance(assessment_scores, dict) else []
    assessment_average = round(sum(_safe_float(v) for v in assessment_values) / max(1, len(assessment_values)), 2) if assessment_values else 0.0

    profile_views = _safe_int(signals.get("profile_views_received_30d"), 0)
    saved_by_recruiters = _safe_int(signals.get("saved_by_recruiters_30d"), 0)
    search_appearance = _safe_int(signals.get("search_appearance_30d"), 0)

    views_score = min(100.0, math.log1p(profile_views) / math.log1p(1000) * 100.0) if profile_views else 0.0
    saved_score = min(100.0, math.log1p(saved_by_recruiters) / math.log1p(100) * 100.0) if saved_by_recruiters else 0.0
    search_score = min(100.0, math.log1p(search_appearance) / math.log1p(1000) * 100.0) if search_appearance else 0.0
    recruiter_interest_score = round((views_score * 0.45) + (saved_score * 0.35) + (search_score * 0.20), 2)

    open_to_work = bool(signals.get("open_to_work_flag"))
    notice_period = _safe_int(signals.get("notice_period_days"), 180)
    availability_score = 100.0 if open_to_work and notice_period <= 30 else max(0.0, 100.0 - (notice_period * 1.2))
    availability_score = round(min(100.0, max(0.0, availability_score)), 2)

    return {
        "profile_completeness_score": profile_completeness_score,
        "recruiter_response_rate": recruiter_response_rate,
        "github_activity_score": github_activity_score,
        "assessment_average": assessment_average,
        "recruiter_interest_score": recruiter_interest_score,
        "availability_score": availability_score,
    }


def build_feature_dataset(input_path: Optional[Path | str] = None, output_path: Optional[Path | str] = None, batch_size: int = 5000) -> pd.DataFrame:
    configure_logging()
    dataset_path = Path(input_path) if input_path else DATASET_PATH
    output_file = Path(output_path) if output_path else OUTPUT_PATH

    if not dataset_path.exists():
        raise FileNotFoundError(f"Candidate dataset not found at {dataset_path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    total_candidates = 0

    with dataset_path.open("r", encoding="utf-8") as handle:
        batch: List[Dict[str, Any]] = []
        for line_number, line in enumerate(handle, start=1):
            candidate_line = line.strip()
            if not candidate_line:
                continue

            candidate = json.loads(candidate_line)
            batch.append(candidate)
            total_candidates += 1

            if len(batch) >= batch_size:
                batch_rows = []
                for candidate in batch:
                    semantic_text = _extract_semantic_text(candidate)
                    ai_capabilities = {name: _score_ai_capability(candidate, name) for name in AI_TERMS}
                    career_evidence = _score_career_evidence(candidate)
                    authenticity = _score_authenticity(candidate, ai_capabilities)
                    behavior = _score_behavioral_features(candidate)

                    batch_rows.append(
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "semantic_text": semantic_text,
                            "llm_experience_score": ai_capabilities["llm"],
                            "ml_experience_score": ai_capabilities["ml"],
                            "deep_learning_score": ai_capabilities["deep_learning"],
                            "rag_vector_database_score": ai_capabilities["rag_vector_database"],
                            "mlops_score": ai_capabilities["mlops"],
                            **career_evidence,
                            **authenticity,
                            **behavior,
                        }
                    )
                rows.extend(batch_rows)
                logger.info("Processed %s candidates", total_candidates)
                batch = []

        if batch:
            batch_rows = []
            for candidate in batch:
                semantic_text = _extract_semantic_text(candidate)
                ai_capabilities = {name: _score_ai_capability(candidate, name) for name in AI_TERMS}
                career_evidence = _score_career_evidence(candidate)
                authenticity = _score_authenticity(candidate, ai_capabilities)
                behavior = _score_behavioral_features(candidate)
                batch_rows.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "semantic_text": semantic_text,
                        "llm_experience_score": ai_capabilities["llm"],
                        "ml_experience_score": ai_capabilities["ml"],
                        "deep_learning_score": ai_capabilities["deep_learning"],
                        "rag_vector_database_score": ai_capabilities["rag_vector_database"],
                        "mlops_score": ai_capabilities["mlops"],
                        **career_evidence,
                        **authenticity,
                        **behavior,
                    }
                )
            rows.extend(batch_rows)

    feature_frame = pd.DataFrame(rows)
    feature_frame.to_parquet(output_file, index=False)
    logger.info("Saved %s rows to %s", len(feature_frame), output_file)
    return feature_frame


if __name__ == "__main__":
    df = build_feature_dataset()
    print(f"Feature dataframe shape: {df.shape}")
    print(df.head(3).to_string(index=False))
