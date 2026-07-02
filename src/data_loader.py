from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "data" / "raw" / "extracted" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
SCHEMA_PATH = ROOT_DIR / "data" / "raw" / "extracted" / "India_runs_data_and_ai_challenge" / "candidate_schema.json"


def _resolve_path(path: Optional[Path | str]) -> Path:
    if path is None:
        return DATASET_PATH
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (ROOT_DIR / candidate).resolve()
    return candidate


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _extract_field_paths(node: Dict[str, Any], prefix: str = "") -> List[str]:
    field_paths: List[str] = []
    node_type = node.get("type")

    if node_type == "object":
        for prop_name, prop_schema in node.get("properties", {}).items():
            child_prefix = f"{prefix}.{prop_name}" if prefix else prop_name
            field_paths.extend(_extract_field_paths(prop_schema, child_prefix))
        return field_paths

    if node_type == "array":
        field_paths.append(prefix or "[]")
        return field_paths

    field_paths.append(prefix)
    return field_paths


def _build_nested_summary(node: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    node_type = node.get("type")
    if node_type == "object":
        properties = {
            name: _build_nested_summary(prop_schema, f"{prefix}.{name}" if prefix else name)
            for name, prop_schema in node.get("properties", {}).items()
        }
        return {"type": "object", "required": node.get("required", []), "properties": properties}

    if node_type == "array":
        item_schema = node.get("items", {})
        return {"type": "array", "items": _build_nested_summary(item_schema, f"{prefix}[]" if prefix else "[]")}

    summary: Dict[str, Any] = {"type": node_type}
    if "enum" in node:
        summary["enum"] = node["enum"]
    if "format" in node:
        summary["format"] = node["format"]
    return summary


def load_candidates(path: Optional[Path | str] = None, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Stream candidates from the JSONL file without loading the full dataset into memory."""
    dataset_path = _resolve_path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Candidate dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        seen = 0
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            seen += 1
            if limit is not None and seen >= limit:
                break


def inspect_schema(schema_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load the candidate schema and summarize its structure."""
    schema_file = _resolve_path(schema_path or SCHEMA_PATH)
    with schema_file.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    top_level_fields = list(schema.get("properties", {}).keys())
    nested_structure = _build_nested_summary(schema)
    field_paths = _extract_field_paths(schema)

    return {
        "schema_path": str(schema_file),
        "top_level_fields": top_level_fields,
        "required_fields": schema.get("required", []),
        "field_paths": field_paths,
        "nested_structure": nested_structure,
    }


def get_sample_candidate(path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Return the first candidate record from the JSONL file."""
    candidates = load_candidates(path=path, limit=1)
    try:
        return next(candidates)
    except StopIteration as exc:
        raise ValueError("No candidates were found in the dataset") from exc


def get_dataset_statistics(path: Optional[Path | str] = None, schema_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Compute lightweight statistics over the candidate dataset using a single streaming pass."""
    dataset_path = _resolve_path(path)
    schema_info = inspect_schema(schema_path)
    field_paths = schema_info["field_paths"]

    candidate_count = 0
    available_fields: set[str] = set()
    missing_counts = {field_path: 0 for field_path in field_paths}

    for candidate in load_candidates(dataset_path):
        candidate_count += 1
        available_fields.update(candidate.keys())

        for field_path in field_paths:
            if field_path in {"[]", ""}:
                continue
            current_value: Any = candidate
            for part in field_path.split("."):
                if isinstance(current_value, dict) and part in current_value:
                    current_value = current_value[part]
                else:
                    current_value = None
                    break
            if _is_missing(current_value):
                missing_counts[field_path] += 1

    missing_percentage = {
        field_path: round((missing_counts[field_path] / candidate_count) * 100, 2) if candidate_count else 0.0
        for field_path in field_paths
    }

    return {
        "candidate_count": candidate_count,
        "available_fields": sorted(available_fields),
        "nested_structure": schema_info["nested_structure"],
        "missing_values_percentage": missing_percentage,
    }


if __name__ == "__main__":
    stats = get_dataset_statistics()
    sample_candidate = get_sample_candidate()

    print(f"Number of candidates: {stats['candidate_count']}")
    print("Available fields:")
    for field in stats["available_fields"]:
        print(f"- {field}")

    print("Nested structure:")
    print(json.dumps(stats["nested_structure"], indent=2, ensure_ascii=False))

    print("Missing values percentage:")
    for field, percentage in stats["missing_values_percentage"].items():
        print(f"- {field}: {percentage:.2f}%")

    print("Sample candidate:")
    print(json.dumps(sample_candidate, indent=2, ensure_ascii=False))
