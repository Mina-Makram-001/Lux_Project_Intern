# clean_lob_status.py
from rapidfuzz import process, fuzz


def _normalize(text):
    return str(text).strip().upper().replace("_", " ")


def fuzzy_map_column(series, choices, threshold=75):
    """
    Fuzzy-matches each value in `series` against `choices`,
    ignoring case/underscore differences, returning the clean value.
    """
    choices = [str(c) for c in choices]
    normalized_choices = [_normalize(c) for c in choices]

    mapping = {}
    for val in series.dropna().unique():
        result = process.extractOne(
            _normalize(val), normalized_choices, scorer=fuzz.WRatio
        )
        if result and result[1] >= threshold:
            _, score, idx = result
            mapping[val] = choices[idx]  # return the ORIGINAL clean value
        else:
            mapping[val] = val  # keep original if no good match

    return series.map(mapping).fillna(series)


def clean_lob_status(df1, lob_col, status_col, lob_choices, status_choices, threshold=75):
    df1[lob_col] = fuzzy_map_column(df1[lob_col], lob_choices, threshold)
    df1[status_col] = fuzzy_map_column(df1[status_col], status_choices, threshold)
    return df1