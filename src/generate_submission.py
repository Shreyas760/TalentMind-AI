from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features.parquet"
EMBEDDINGS_PATH = ROOT_DIR / "data" / "processed" / "candidate_embeddings.npy"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "final_submission.csv"
DEFAULT_JOB_DESCRIPTION = (
    "Senior data scientist with strong experience in machine learning, LLMs, deep learning, "
    "and production AI systems"
)

logger = logging.getLogger("generate_submission")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def validate_submission_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("No ranked candidates were generated")

    if "candidate_id" not in frame.columns:
        raise ValueError("Submission frame must contain 'candidate_id'")

    if frame["candidate_id"].duplicated().any():
        duplicates = frame.loc[frame["candidate_id"].duplicated(), "candidate_id"].tolist()
        raise ValueError(f"Duplicate candidate IDs detected: {duplicates[:10]}")

    if frame["rank"].min() != 1:
        raise ValueError("Ranking must start at 1")

    if not frame["rank"].is_monotonic_increasing:
        raise ValueError("Ranks must be sequential and increasing")

    if not frame["final_score"].is_monotonic_decreasing:
        raise ValueError("Submission must be sorted by final_score descending")


def generate_submission(
    job_description: str = DEFAULT_JOB_DESCRIPTION,
    features_path: Optional[Path | str] = None,
    embeddings_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    configure_logging()

    features_file = Path(features_path) if features_path else FEATURES_PATH
    embeddings_file = Path(embeddings_path) if embeddings_path else EMBEDDINGS_PATH
    output_file = Path(output_path) if output_path else OUTPUT_PATH

    if not features_file.exists():
        raise FileNotFoundError(f"Feature parquet file not found: {features_file}")
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {embeddings_file}")

    logger.info("Loading features from %s", features_file)
    logger.info("Loading embeddings from %s", embeddings_file)

    from ranking_model import rank_candidates
    from explainability import explain_shortlist

    ranked = rank_candidates(
        job_description=job_description,
        features_path=features_file,
        embeddings_path=embeddings_file,
        top_n=top_n,
    )

    logger.info("Generating explanations for %s ranked candidates", len(ranked))
    feature_frame = pd.read_parquet(features_file)
    explanations = explain_shortlist(ranked_candidates=ranked, feature_frame=feature_frame)

    explanation_frame = pd.DataFrame(explanations)
    if "confidence" not in explanation_frame.columns or "summary" not in explanation_frame.columns:
        raise ValueError("Explanation payload is missing required columns")

    submission = ranked.merge(
        explanation_frame[["candidate_id", "confidence", "summary"]],
        on="candidate_id",
        how="left",
    )
    submission = submission[["candidate_id", "final_score", "confidence", "summary"]].copy()
    submission = submission.sort_values("final_score", ascending=False).reset_index(drop=True)
    submission.insert(1, "rank", range(1, len(submission) + 1))
    submission = submission[["candidate_id", "rank", "final_score", "confidence", "summary"]]

    validate_submission_frame(submission)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_file, index=False)
    logger.info("Saved %s ranked submissions to %s", len(submission), output_file)

    print("Top 20 candidates:")
    print(submission.head(20).to_string(index=False))
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a ranked submission CSV for candidate shortlist")
    parser.add_argument("--job-description", default=DEFAULT_JOB_DESCRIPTION, help="Job description text")
    parser.add_argument("--features-path", default=str(FEATURES_PATH), help="Path to processed feature parquet")
    parser.add_argument("--embeddings-path", default=str(EMBEDDINGS_PATH), help="Path to candidate embedding npy")
    parser.add_argument("--output-path", default=str(OUTPUT_PATH), help="Destination CSV for the submission")
    parser.add_argument("--top-n", type=int, default=None, help="Optional limit on number of ranked candidates")
    args = parser.parse_args()

    generate_submission(
        job_description=args.job_description,
        features_path=args.features_path,
        embeddings_path=args.embeddings_path,
        output_path=args.output_path,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
