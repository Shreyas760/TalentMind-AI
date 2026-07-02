from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
import torch

from src.resume_intelligence import add_resume_intelligence_scores

ROOT_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features.parquet"
EMBEDDINGS_PATH = ROOT_DIR / "data" / "processed" / "candidate_embeddings.npy"

logger = logging.getLogger("ranking_model")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_features(features_path: Optional[Path | str] = None) -> pd.DataFrame:
    features_file = Path(features_path) if features_path else FEATURES_PATH
    if not features_file.exists():
        raise FileNotFoundError(f"Feature parquet file not found: {features_file}")
    logger.info("Loading engineered features from %s", features_file)
    return pd.read_parquet(features_file)


def _load_embeddings(embeddings_path: Optional[Path | str] = None) -> Dict[str, Any]:
    embeddings_file = Path(embeddings_path) if embeddings_path else EMBEDDINGS_PATH
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {embeddings_file}")
    logger.info("Loading candidate embeddings from %s", embeddings_file)
    payload = np.load(embeddings_file, allow_pickle=True).item()
    return {"candidate_ids": payload["candidate_ids"], "embeddings": payload["embeddings"]}


def _load_job_embedding(job_description: str, model: Optional[SentenceTransformer] = None) -> np.ndarray:
    if not job_description or not str(job_description).strip():
        raise ValueError("job_description must be a non-empty string")

    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    with torch.inference_mode():
        embedding = model.encode(
            [job_description],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return embedding.astype(np.float32)


def rank_candidates(
    job_description: str,
    features_path: Optional[Path | str] = None,
    embeddings_path: Optional[Path | str] = None,
    top_n: int = 100,
) -> pd.DataFrame:
    configure_logging()
    features = _load_features(features_path)
    embeddings_payload = _load_embeddings(embeddings_path)

    if "candidate_id" not in features.columns:
        raise ValueError("The feature dataframe must contain a 'candidate_id' column")

    candidate_ids = np.asarray(embeddings_payload["candidate_ids"], dtype=object)
    embeddings = np.asarray(embeddings_payload["embeddings"], dtype=np.float32)

    if len(candidate_ids) != len(features):
        raise ValueError(
            f"Candidate count mismatch: features={len(features)}, embeddings={len(candidate_ids)}"
        )

    if candidate_ids.shape[0] == 0:
        raise ValueError("No candidates available for ranking")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    job_embedding = _load_job_embedding(job_description, model=model)

    logger.info("Computing semantic similarity against %s candidates", len(candidate_ids))
    semantic_similarity = (embeddings @ job_embedding.T).reshape(-1)
    semantic_similarity = np.clip(semantic_similarity.astype(np.float32), -1.0, 1.0)

    feature_frame = features.copy()
    feature_frame["semantic_similarity"] = semantic_similarity

    feature_groups = [
        "llm_experience_score",
        "ml_experience_score",
        "deep_learning_score",
        "rag_vector_database_score",
        "mlops_score",
    ]
    authenticity_groups = [
        "authenticity_score",
        "skill_evidence_alignment",
        "assessment_alignment",
        "github_activity_factor",
    ]
    career_groups = [
        "production_ai_experience_score",
        "career_transition_score",
        "ownership_score",
    ]
    behavior_groups = [
        "recruiter_interest_score",
        "recruiter_response_rate",
        "profile_completeness_score",
    ]

    scaler = MinMaxScaler()
    selected_features = [
        "semantic_similarity",
        *feature_groups,
        *authenticity_groups,
        *career_groups,
        *behavior_groups,
    ]

    feature_matrix = feature_frame[selected_features].astype(float)
    normalized_matrix = scaler.fit_transform(feature_matrix)
    normalized_frame = pd.DataFrame(normalized_matrix, columns=selected_features, index=feature_frame.index)

    ai_capability_average = normalized_frame[feature_groups].mean(axis=1)
    authenticity_average = normalized_frame[authenticity_groups].mean(axis=1)
    career_average = normalized_frame[career_groups].mean(axis=1)
    behavior_average = normalized_frame[behavior_groups].mean(axis=1)

    feature_frame["ai_capability_average"] = ai_capability_average
    feature_frame["authenticity_average"] = authenticity_average
    feature_frame["career_average"] = career_average
    feature_frame["behavior_average"] = behavior_average

    feature_frame["final_score"] = (
        0.45 * normalized_frame["semantic_similarity"]
        + 0.20 * normalized_frame["authenticity_score"]
        + 0.15 * feature_frame["ai_capability_average"]
        + 0.10 * normalized_frame["production_ai_experience_score"]
        + 0.05 * normalized_frame["career_transition_score"]
        + 0.05 * normalized_frame["recruiter_interest_score"]
    )

    feature_frame = add_resume_intelligence_scores(feature_frame)
    feature_frame["final_score"] = feature_frame["final_score"] - (
        0.03 * feature_frame["resume_inflation_score"] / 100.0
    )

    ranked = feature_frame[["candidate_id", "semantic_similarity", "final_score", "resume_inflation_score", "resume_credibility_score", "evidence_coverage_pct"]].copy()
    ranked["semantic_similarity"] = ranked["semantic_similarity"].astype(float)
    ranked["final_score"] = ranked["final_score"].astype(float)
    ranked = ranked.sort_values(["final_score", "semantic_similarity"], ascending=False).reset_index(drop=True)

    if top_n is not None:
        ranked = ranked.head(int(top_n))

    logger.info("Generated rankings for %s candidates", len(ranked))
    return ranked


if __name__ == "__main__":
    ranked = rank_candidates(
        "Senior data scientist with strong experience in machine learning, LLMs, deep learning, and production AI systems",
        top_n=10,
    )
    print(ranked.to_string(index=False))
