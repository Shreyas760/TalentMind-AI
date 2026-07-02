# TalentMind AI

TalentMind AI is an intelligent candidate ranking and resume credibility platform designed to help recruiters and hiring teams identify strong-fit candidates from large talent pools. The system combines semantic matching, feature-based ranking, explainability, and resume inflation detection to surface candidates who are both relevant and credible.

![Screenshots Placeholder](docs/screenshots/placeholder.png)

## Project Overview

TalentMind AI addresses a common hiring challenge: ranking candidates efficiently while reducing the risk of overvaluing inflated or inconsistent resumes. The platform ingests a large pool of candidate profiles, generates rich features from career and skill signals, computes semantic similarity to a target role, and produces a ranked shortlist with recruiter-friendly explanations.

The solution is built as a modular Python pipeline with:
- robust data loading and feature engineering
- semantic embeddings for role matching
- hybrid ranking logic
- explainable candidate rationale
- resume credibility and inflation analysis
- an interactive Streamlit interface for end-to-end exploration

## Problem Statement

Recruiters often need to review thousands of candidate profiles manually. Traditional keyword matching can overlook strong candidates, while weak or inflated resumes can be over-prioritized. TalentMind AI tackles this by combining:
- semantic understanding of job descriptions and candidate profiles
- evidence-based scoring of AI and engineering experience
- credibility detection for inflated or inconsistent claims
- transparent explanations for shortlist decisions

## Architecture Diagram

```text
+----------------------+        +---------------------------+
|  Candidate Dataset   |        |  Job Description Input    |
|  (JSONL / Processed) |        |  (Text)                   |
+----------+-----------+        +------------+--------------+
           |                                  |
           v                                  v
+----------------------+          +---------------------------+
| Data Loader /       |          | Embedding Generation      |
| Feature Engineering |          | (SentenceTransformers)   |
+----------+-----------+          +------------+--------------+
           |                                  |
           +-------------------+----------------+
                               |
                               v
+------------------------------+------------------------------+
| Engineered Feature Matrix                                |
| - AI capability scores                                   |
| - Career evidence scores                                 |
| - Authenticity signals                                   |
| - Behavioral signals                                     |
+------------------------------+------------------------------+
                               |
                               v
+------------------------------+------------------------------+
| Hybrid Ranking Engine                                     |
| - Semantic similarity                                     |
| - Feature-based scoring                                   |
| - Resume intelligence penalty                             |
+------------------------------+------------------------------+
                               |
                               v
+------------------------------+------------------------------+
| Explainability Layer                                      |
| - Strengths / Weaknesses / Risk Flags                     |
| - Confidence / Why Selected                              |
+------------------------------+------------------------------+
                               |
                               v
+------------------------------+------------------------------+
| Streamlit App / Submission Export                         |
| - Interactive UI                                         |
| - Top-K ranking                                           |
| - CSV download                                            |
+-----------------------------------------------------------+
```

## Pipeline

1. Load candidate data from the provided dataset.
2. Generate structured features from resumes, career history, and signals.
3. Create semantic embeddings for candidates and job descriptions.
4. Rank candidates using a hybrid scoring approach.
5. Detect resume inflation and credibility concerns.
6. Generate recruiter-readable explanations.
7. Export a final submission CSV and surface results through Streamlit.

## Dataset

The project uses the provided candidate dataset bundled under the data directory. The pipeline expects:
- candidate records in JSONL format
- processed feature parquet files
- precomputed candidate embeddings

The main data flow is:
- raw data: data/raw/extracted/
- processed data: data/processed/
- generated outputs: outputs/

## Feature Engineering

The feature engineering module transforms raw candidate data into a structured set of signals, including:
- AI capability scores for LLMs, ML, deep learning, RAG/vector databases, and MLOps
- career evidence and ownership scores
- authenticity and credibility-related features
- recruiter engagement and profile completeness metrics
- semantic text representation for matching

These features form the backbone of the ranking engine and credibility model.

## Hybrid Ranking Model

The ranking model combines:
- semantic similarity from sentence embeddings
- engineered feature scores
- credibility-aware adjustments to reduce the impact of inflated profiles

The final ranking is designed to balance relevance, evidence strength, and recruiter signal quality.

## Explainability Engine

Each shortlisted candidate receives an explanation package containing:
- strengths
- weaknesses
- risk flags
- confidence score
- a concise summary of why the candidate was selected

This makes the shortlist more transparent and better suited to recruiter review.

## Resume Intelligence

The resume intelligence layer detects possible inflation and inconsistency by evaluating:
- claimed advanced AI capabilities such as LLMs, RAG, fine-tuning, and PyTorch
- consistency between claimed skills and career evidence
- assessment outcomes versus stated expertise
- GitHub activity versus advanced skill claims
- broader career progression and deployment ownership

This produces:
- Skill Inflation Score
- Resume Inflation Score
- Resume Credibility Score
- Evidence Coverage %

## Tech Stack

- Python 3.10+
- pandas
- NumPy
- scikit-learn
- sentence-transformers
- PyTorch
- Plotly
- Streamlit
- pyarrow

## Performance

The pipeline is designed to work efficiently on large candidate sets through:
- batch processing for feature generation
- vectorized scoring operations
- embedding-based ranking without per-candidate loops where possible
- lightweight explainability generation for shortlisted candidates

## Future Improvements

Planned enhancements include:
- better calibration of ranking weights using labeled recruiter feedback
- richer resume parsing for structured evidence extraction
- support for multi-role and multi-location hiring workflows
- integration with applicant tracking systems (ATS)
- more advanced explainability with citation-style evidence
- deployment to a cloud-hosted web application

## Execution Steps

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing pipeline:
   ```bash
   python src/feature_engineering.py
   python src/embeddings.py
   ```
4. Generate the final submission:
   ```bash
   python src/generate_submission.py
   ```
5. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Screenshots Placeholder

Add screenshots of the app UI and candidate analysis view here in future iterations.

## Team Credits

Built as a hackathon project by the TalentMind AI team.
