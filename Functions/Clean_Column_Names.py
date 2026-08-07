import pandas as pd
from rapidfuzz import process, fuzz
import re

SYNONYMS = {
    "buisness": "lob",
    "business": "lob",
    "accident date": "loss date",
    "accident": "loss",
}

def preprocess_header(name):
    """Normalizes string and applies insurance domain synonyms (word-boundary safe)."""
    cleaned = str(name).replace('_', ' ').lower().strip()
    for word, synonym in SYNONYMS.items():
        cleaned = re.sub(rf'\b{re.escape(word)}\b', synonym, cleaned)
    return cleaned

def clean_and_align_columns(df_data, df_target, score_cutoff=60.0, scorer=fuzz.token_sort_ratio):
    """
    Automates matching source columns to target schema columns using RapidFuzz.
    Uses global best-match assignment instead of first-come-first-served,
    so the highest-confidence pairings win regardless of column order.
    """
    source_columns = df_data.columns.str.strip().tolist()

    target_col_name = df_target.columns[0]
    target_columns = df_target[target_col_name].dropna().astype(str).str.strip().tolist()

    if len(set(source_columns)) != len(source_columns):
        raise ValueError("df_data has duplicate column names; resolve before matching.")
    if len(set(target_columns)) != len(target_columns):
        raise ValueError("Target schema has duplicate column names; resolve before matching.")

    cleaned_sources = [preprocess_header(c) for c in source_columns]
    cleaned_targets = [preprocess_header(c) for c in target_columns]

    # 1. Collect ALL candidate matches above cutoff (not just each source's top pick)
    candidates = []  # (score, orig_source, matched_target)
    for orig_source, clean_source in zip(source_columns, cleaned_sources):
        matches = process.extract(
            clean_source,
            cleaned_targets,
            scorer=scorer,
            score_cutoff=score_cutoff,
            limit=None,
        )
        for _, score, idx in matches:
            candidates.append((score, orig_source, target_columns[idx]))

    # 2. Assign globally best-first, so a great match can't be blocked
    #    by a weaker match that merely came first in column order
    candidates.sort(key=lambda x: x[0], reverse=True)

    column_mapping = {}
    used_sources = set()
    used_targets = set()
    for score, src, tgt in candidates:
        if src in used_sources or tgt in used_targets:
            continue
        column_mapping[src] = tgt
        used_sources.add(src)
        used_targets.add(tgt)

    unmapped_sources = [c for c in source_columns if c not in column_mapping]
    if unmapped_sources:
        print(f"[clean_and_align_columns] Unmapped source columns (dropped): {unmapped_sources}")

    df_cleaned = df_data.rename(columns=column_mapping)

    for col in target_columns:
        if col not in df_cleaned.columns:
            df_cleaned[col] = None

    df_cleaned = df_cleaned[target_columns]

    return df_cleaned