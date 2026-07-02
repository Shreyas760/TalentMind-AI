from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from embeddings import calculate_similarity


class JobMatcher:
    def __init__(self, embeddings_path: Optional[Path | str] = None) -> None:
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None

    def match(self, job_description: str, top_n: int = 100) -> pd.DataFrame:
        if not job_description or not job_description.strip():
            raise ValueError("job_description must be a non-empty string")
        return calculate_similarity(job_description, embeddings_path=self.embeddings_path, top_n=top_n)


if __name__ == "__main__":
    matcher = JobMatcher()
    description = "Senior data scientist with strong ML, deep learning, and production AI experience"
    results = matcher.match(description, top_n=10)
    print(results.to_string(index=False))
