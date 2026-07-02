from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features.parquet"
EMBEDDINGS_PATH = ROOT_DIR / "data" / "processed" / "candidate_embeddings.npy"

logger = logging.getLogger("embeddings")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_embedding_model(device: Optional[str] = None) -> Tuple[SentenceTransformer, str]:
    """Load the sentence-transformers embedding model and select CPU/GPU automatically."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading embedding model on device: %s", device)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    return model, device


def generate_candidate_embeddings(
    input_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    batch_size: int = 2048,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Generate embeddings for all candidate semantic_text values in batches and save them to disk."""
    configure_logging()
    input_file = Path(input_path) if input_path else FEATURES_PATH
    output_file = Path(output_path) if output_path else EMBEDDINGS_PATH

    if not input_file.exists():
        raise FileNotFoundError(f"Feature parquet file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading feature parquet from %s", input_file)
    feature_frame = pd.read_parquet(input_file)
    if "semantic_text" not in feature_frame.columns:
        raise ValueError("The input parquet file does not contain a 'semantic_text' column")

    texts = feature_frame["semantic_text"].fillna("").astype(str).tolist()
    candidate_ids = feature_frame["candidate_id"].astype(str).tolist()

    model, device = load_embedding_model()

    if not texts:
        raise ValueError("No semantic_text values were found in the feature parquet file")

    sample_embedding = model.encode([texts[0]], convert_to_numpy=True, normalize_embeddings=True)
    embedding_dim = sample_embedding.shape[1]

    temp_memmap_path = output_file.with_suffix(".mmap")
    if temp_memmap_path.exists():
        temp_memmap_path.unlink()

    embeddings_memmap = np.memmap(
        temp_memmap_path,
        dtype="float32",
        mode="w+",
        shape=(len(texts), embedding_dim),
    )

    logger.info("Generating %s embeddings with batch size %s", len(texts), batch_size)
    for start in tqdm(range(0, len(texts), batch_size), disable=not show_progress, desc="Embedding candidates"):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        with torch.inference_mode():
            batch_embeddings = model.encode(
                batch_texts,
                batch_size=min(batch_size, 256),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        embeddings_memmap[start:end] = batch_embeddings.astype("float32")

    embeddings_memmap.flush()
    embeddings = np.array(embeddings_memmap, dtype="float32", copy=False)
    payload = {"candidate_ids": np.array(candidate_ids, dtype=object), "embeddings": embeddings}
    np.save(output_file, payload, allow_pickle=True)

    del embeddings_memmap
    if temp_memmap_path.exists():
        try:
            temp_memmap_path.unlink()
        except PermissionError:
            logger.warning("Temporary memmap file is still locked; leaving it on disk")

    logger.info("Saved embeddings to %s", output_file)
    return {"output_path": str(output_file), "candidate_ids": candidate_ids, "embeddings": embeddings}


def calculate_similarity(
    job_description: str,
    embeddings_path: Optional[Path | str] = None,
    top_n: int = 100,
) -> pd.DataFrame:
    """Create a job embedding and compute similarity against all candidate embeddings."""
    configure_logging()
    embedding_file = Path(embeddings_path) if embeddings_path else EMBEDDINGS_PATH
    if not embedding_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_file}")

    payload = np.load(embedding_file, allow_pickle=True).item()
    candidate_ids = payload["candidate_ids"]
    embeddings = payload["embeddings"]

    model, _ = load_embedding_model()
    query_embedding = model.encode([job_description], convert_to_numpy=True, normalize_embeddings=True)
    similarities = (embeddings @ query_embedding.T).reshape(-1)
    similarity_scores = np.clip(similarities, -1.0, 1.0)

    ranked_indices = np.argsort(similarity_scores)[::-1][:top_n]
    ranked_df = pd.DataFrame(
        {
            "candidate_id": candidate_ids[ranked_indices],
            "similarity_score": similarity_scores[ranked_indices],
        }
    )
    ranked_df = ranked_df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    return ranked_df


if __name__ == "__main__":
    result = generate_candidate_embeddings()
    print(f"Saved {len(result['candidate_ids'])} candidate embeddings to {result['output_path']}")
    job_matches = calculate_similarity("Data scientist with experience in machine learning, LLMs, and production AI systems")
    print(job_matches.head(10).to_string(index=False))
