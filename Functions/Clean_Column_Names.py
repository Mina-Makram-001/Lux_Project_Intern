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
    Each target name can only be used once.
    """
    # working copy of the pool -- entries get removed as they're claimed
    pool = {c: c.replace("LA_", "") for c in to_list}
    from_list = [str(c).upper().replace(" ", "_") for c in df.columns]

    new_columns = []
    for col in from_list:
        if col in synonyms and synonyms[col] in pool:
            target = synonyms[col]
            new_columns.append(target)
            del pool[target]  # remove so it can't be matched again
            continue

        if not pool:
            new_columns.append(col)  # nothing left to match against
            continue

        best_match, score, match_key = process.extractOne(col, pool, scorer=fuzz.WRatio)
        new_columns.append(match_key)
        del pool[match_key]  # remove so it can't be matched again

    df.columns = new_columns
    return df