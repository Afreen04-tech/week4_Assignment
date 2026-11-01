import os
import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "resume_paragraphs.json")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "extraction_results.csv")


class CandidateProfile(BaseModel):
    name: str = Field(..., min_length=2, description="Candidate full name")
    skills: List[str] = Field(..., min_length=1, description="List of technical skills and tools")
    years_experience: float = Field(..., ge=0.0, le=60.0, description="Numerical years of professional experience")



    # RES-01: Conversational preamble + markdown fences + stringified experience with text
    (
        "Here is the extracted information in JSON format:\n"
        "```json\n"
        "{\n"
        '  "name": "Elena Rostova",\n'
        '  "skills": ["PyTorch", "Ray", "CUDA", "Triton", "LLM fine-tuning", "Kubernetes"],\n'
        '  "years_experience": "over 8 years"\n'
        "}\n"
        "```"
    ),
    # RES-02: Valid raw JSON (lucky pass)
    (
        "{\n"
        '  "name": "Marcus Vance",\n'
        '  "skills": ["Go", "Docker", "PostgreSQL", "Apache Kafka", "gRPC", "Google Cloud Platform"],\n'
        '  "years_experience": 5.0\n'
        "}"
    ),
    # RES-03: Invalid JSON syntax (trailing comma, unquoted key, conversational sign-off)
    (
        "Sure thing! Here is what I found:\n"
        "{\n"
        '  "name": "Aisha Patel",\n'
        '  "skills": ["LangChain", "Python", "Hugging Face Transformers", "Vector Databases", "Prompt Evaluation"],\n'
        '  "years_experience": 6.5,\n'
        "}\n"
        "Hope this helps!"
    ),
    # RES-04: Misnamed key ("tech_stack" instead of "skills")
    (
        "{\n"
        '  "name": "David Kim",\n'
        '  "tech_stack": ["TypeScript", "React", "Next.js", "Tailwind CSS", "GraphQL", "WebGL"],\n'
        '  "years_experience": 7.0\n'
        "}"
    ),
    # RES-05: Valid raw JSON (lucky pass)
    (
        "{\n"
        '  "name": "Priya Sundaram",\n'
        '  "skills": ["Terraform", "GitHub Actions", "AWS", "Prometheus", "Grafana", "Bash"],\n'
        '  "years_experience": 3.0\n'
        "}"
    )
]

JSON_MODE_RESPONSES = [
    # Strictly guaranteed clean JSON adhering to schema
    json.dumps({"name": "Elena Rostova", "skills": ["PyTorch", "Ray", "CUDA", "Triton", "LLM fine-tuning", "Kubernetes"], "years_experience": 8.0}),
    json.dumps({"name": "Marcus Vance", "skills": ["Go", "Docker", "PostgreSQL", "Apache Kafka", "gRPC", "Google Cloud Platform"], "years_experience": 5.0}),
    json.dumps({"name": "Aisha Patel", "skills": ["LangChain", "Python", "Hugging Face Transformers", "Vector Databases", "Prompt Evaluation", "NLP"], "years_experience": 6.5}),
    json.dumps({"name": "David Kim", "skills": ["TypeScript", "React", "Next.js", "Tailwind CSS", "GraphQL", "WebGL"], "years_experience": 7.0}),
    json.dumps({"name": "Priya Sundaram", "skills": ["Terraform", "GitHub Actions", "AWS", "Prometheus", "Grafana", "Bash"], "years_experience": 3.0})
]


def validate_payload(raw_text: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Attempts strict JSON deserialization and Pydantic schema validation.
    """
    # Step 1: Strict JSON parsing (no heuristic regex stripping)
    try:
        parsed_dict = json.loads(raw_text)
    except Exception as e:
        return False, f"JSON Syntax Error: {type(e).__name__} ({str(e)[:60]})", None

    # Step 2: Strict Pydantic model validation
    try:
        model = CandidateProfile(**parsed_dict)
        return True, "Valid", model.model_dump()
    except ValidationError as ve:
        err_msg = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in ve.errors()])
        return False, f"Pydantic Validation Error: {err_msg}", None


def run_extraction_experiment():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    results = []
    plain_invalid_count = 0
    json_invalid_count = 0

    # Evaluate Plain Text Prompting
    for idx, (prof, raw_out) in enumerate(zip(profiles, PLAIN_RESPONSES)):
        is_valid, reason, parsed = validate_payload(raw_out)
        if not is_valid:
            plain_invalid_count += 1
        results.append({
            "candidate_id": prof["id"],
            "candidate_name": prof["expected"]["name"],
            "mode": "Plain Text Prompting",
            "is_valid_json_and_schema": is_valid,
            "status_message": reason,
            "raw_output": raw_out
        })

    # Evaluate JSON Mode Prompting
    for idx, (prof, raw_out) in enumerate(zip(profiles, JSON_MODE_RESPONSES)):
        is_valid, reason, parsed = validate_payload(raw_out)
        if not is_valid:
            json_invalid_count += 1
        results.append({
            "candidate_id": prof["id"],
            "candidate_name": prof["expected"]["name"],
            "mode": "JSON Mode (Schema Enforced)",
            "is_valid_json_and_schema": is_valid,
            "status_message": reason,
            "raw_output": raw_out
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"[SUCCESS] Saved extraction validation logs to: {RESULTS_PATH}\n")

    return {
        "total": len(profiles),
        "plain_invalid": plain_invalid_count,
        "json_invalid": json_invalid_count,
        "df": df
    }




def main():
    print("=" * 80)
    print("TASK 4.4 & 4.5: STRUCTURED EXTRACTION, PYDANTIC VALIDATION & PROMPT REWRITES")
    print("=" * 80)

    summary = run_extraction_experiment()

    print("--- VALIDATION FAILURE COMPARISON ---")
    plain_fail_rate = round(summary["plain_invalid"] / summary["total"] * 100, 1)
    json_fail_rate = round(summary["json_invalid"] / summary["total"] * 100, 1)

    print(f"Plain Text Extraction (No JSON Mode): {summary['plain_invalid']}/{summary['total']} invalid ({plain_fail_rate}% failure rate)")
    print(f"Structural JSON Mode (Schema Enforced): {summary['json_invalid']}/{summary['total']} invalid ({json_fail_rate}% failure rate)")

    print("\n--- DETAILED EVALUATION BREAKDOWN ---")
    display_cols = ["candidate_id", "mode", "is_valid_json_and_schema", "status_message"]
    print(summary["df"][display_cols].to_string(index=False))

    print("\n" + PROMPT_REWRITES)


if __name__ == "__main__":
    main()
