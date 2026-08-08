from rapidfuzz import process, fuzz

SYNONYMS = {
    "AS_OF": "LA_AS_AT_DATE",
    "ACCINDENT_DATE": "LA_LOSS_DATE",
    "PAID_DATE": "LA_PAYMENT_DATE",
    "MAJOR_LINE_OF_BUISNESS": "LA_LOB",
}


def clean_and_align_columns(df, to_list, synonyms=SYNONYMS):
    """
    Renames df's columns to their best-matching name in to_list,
    using manual overrides (synonyms) first, then fuzzy matching.

    Parameters:
    - df: DataFrame whose columns need aligning.
    - to_list: list of target column names (e.g. LA_ names).
    - synonyms: dict of exact overrides for non-fuzzy matches.

    Returns:
    - The same DataFrame with columns renamed in place.
    """
    to_list_stripped = {c: c.replace("LA_", "") for c in to_list}
    from_list = [str(c).upper().replace(" ", "_") for c in df.columns]

    new_columns = []
    for col in from_list:
        if col in synonyms:
            new_columns.append(synonyms[col])
            continue
        best_match, score, _ = process.extractOne(
            col, to_list_stripped, scorer=fuzz.WRatio
        )
        match_key = next(k for k, v in to_list_stripped.items() if v == best_match)
        new_columns.append(match_key)

    df.columns = new_columns
    return df