import pandas as pd

from src.resume_intelligence import add_resume_intelligence_scores


def test_add_resume_intelligence_scores_returns_expected_columns():
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "CAND_1",
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

    scored = add_resume_intelligence_scores(frame)

    assert {"resume_credibility_score", "resume_inflation_score", "evidence_coverage_pct", "skill_inflation_score"}.issubset(scored.columns)
    assert scored.loc[0, "resume_inflation_score"] >= scored.loc[0, "resume_credibility_score"]
    assert 0 <= scored.loc[0, "evidence_coverage_pct"] <= 100
