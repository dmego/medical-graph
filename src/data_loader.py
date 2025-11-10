"""Utility helpers for accessing shared datasets and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_DIR
DATA_DIR = WORKSPACE_ROOT / "data"
REPORT_DIR = WORKSPACE_ROOT / "reports"
OUTPUT_DIR = WORKSPACE_ROOT / "output"

# PageRank directories
PAGERANK_DATA_DIR = OUTPUT_DIR / "data" / "pagerank"
PAGERANK_IMAGE_DIR = OUTPUT_DIR / "images" / "pagerank"

RELATION_TYPES = (
    "acompany_with",
    "belongs_to",
    "common_drug",
    "do_eat",
    "drugs_of",
    "has_symptom",
    "need_check",
    "no_eat",
    "recommand_drug",
    "recommand_eat",
)


def _ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def load_nodes(columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Load node metadata from the shared CSV."""
    path = _ensure_exists(DATA_DIR / "nodes.csv")
    return pd.read_csv(path, usecols=columns)


def load_edges(columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Load edge relations from the shared CSV."""
    path = _ensure_exists(DATA_DIR / "edges.csv")
    return pd.read_csv(path, usecols=columns)


def load_report_json(filename: str) -> Dict:
    """Load precomputed report JSON from the root reports directory."""
    path = _ensure_exists(REPORT_DIR / filename)
    return json.loads(path.read_text(encoding="utf-8"))


def load_report_csv(filename: str, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Load precomputed report CSV from the root reports directory."""
    path = _ensure_exists(REPORT_DIR / filename)
    return pd.read_csv(path, usecols=columns)


def resolve_output(path: str | Path) -> Path:
    """Resolve an output path inside the project outputs directory."""
    absolute = OUTPUT_DIR / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute
