import pandas as pd
import numpy as np


def check_missing(df, col_name): # Check missings in the columns (Main Goal is: Testing Catigories, Loss Date columns)
    """
    Check missings in the columns identified
    """

    if col_name not in df.columns:
        return {
            "column": col_name,
            "status": "ERROR",
            "message": f"Column '{col_name}' not found in dataframe",
            "failed_rows": []
        }

    null_mask = df[col_name].isnull()
    null_count = null_mask.sum()
    failed_rows = df[null_mask].index.tolist()

    return {
        "column": col_name,
        "status": "PASS" if null_count == 0 else "FAIL",
        "message": f"{null_count} nulls found in '{col_name}'" if null_count > 0 else f"No nulls in '{col_name}'",
        "failed_rows": failed_rows
    }


def AS_AT_DATE_CLEAN(df, as_at_col, payment_col):
    df_copy = df.copy()

    df_copy[as_at_col] = pd.to_datetime(df_copy[as_at_col], errors='coerce')
    df_copy[payment_col] = pd.to_datetime(df_copy[payment_col], errors='coerce')

    before_check = check_missing(df_copy, as_at_col)

    if before_check["status"] in ["PASS", "ERROR"]:
        return {
            "column": as_at_col,
            "status": before_check["status"],
            "message": before_check["message"],
            "rows_fixed": 0
        }, df_copy      

    missing_mask = df_copy[as_at_col].isnull()
    quarter_end_dates = df_copy.loc[missing_mask, payment_col].dt.to_period('Q').dt.end_time
    df_copy.loc[missing_mask, as_at_col] = quarter_end_dates

    after_check = check_missing(df_copy, as_at_col)

    return {
        "column": as_at_col,
        "status": "FIXED" if after_check["status"] == "PASS" else "PARTIALLY_FIXED",
        "message": f"{before_check['message']} -> {missing_mask.sum()} rows substituted using quarter-end of '{payment_col}'",
        "rows_fixed": int(missing_mask.sum()),
        "remaining_issues": after_check["failed_rows"]
    }, df_copy


def PAYMENT_DATE_CLEAN(df, payment_date_col, status_col, condition_value='PAID'):
    """
    Wipes payment dates for non-qualifying statuses and reports missing/cleared date counts.
    Parameters: df (DataFrame), payment_date_col (str), status_col (str), condition_value (str, default 'PAID').
    Returns: dict report with keys: column, status, message, wiped_count, missing_paid_dates, df
    """

    df_copy = df.copy()

    df_copy[payment_date_col] = pd.to_datetime(df_copy[payment_date_col], errors='coerce')

    # Step 1: Masks for condition logic
    not_paid_mask = df_copy[status_col] != condition_value
    paid_mask = df_copy[status_col] == condition_value

    # Step 2: Count rows that had a date before wiping them
    wiped_count = int((not_paid_mask & df_copy[payment_date_col].notnull()).sum())

    # Step 3: Wipe payment date where status is NOT the condition value
    df_copy.loc[not_paid_mask, payment_date_col] = np.nan

    # Step 4: Count paid rows missing a payment date
    missing_paid_dates = int(df_copy.loc[paid_mask, payment_date_col].isnull().sum())

    # Step 5: determine overall status for the report
    status = "PASS" if missing_paid_dates == 0 else "FAIL"

    return {
        "column": payment_date_col,
        "status": status,
        "message": f"{wiped_count} dates wiped (status != '{condition_value}'); "
                   f"{missing_paid_dates} PAID rows still missing a payment date",
        "wiped_count": wiped_count,
        "missing_paid_dates": missing_paid_dates,
    }, df_copy


def PAYMENT_VS_LOSS_DATE_VALIDATE(df, payment_date_col, loss_date_col, allow_same_day=True):
    """
    Validates that payment dates occur on or after loss dates and tags each row with validation results.
    Parameters: df (DataFrame), payment_date_col (str), loss_date_col (str), allow_same_day (bool, default True).
    Returns: dict report with keys: column, status, message, checked_count, skipped_count,
    invalid_count, df, invalid_rows
    """

    df_copy = df.copy()

    # Step 1: make sure both columns are actual datetimes (not strings/objects)
    df_copy[payment_date_col] = pd.to_datetime(df_copy[payment_date_col], errors='coerce')
    df_copy[loss_date_col] = pd.to_datetime(df_copy[loss_date_col], errors='coerce')

    # Step 2: only rows where BOTH dates exist can be validated
    both_present_mask = df_copy[payment_date_col].notnull() & df_copy[loss_date_col].notnull()

    # Step 3: define what "invalid" means
    if allow_same_day:
        invalid_mask = both_present_mask & (df_copy[payment_date_col] < df_copy[loss_date_col])
    else:
        invalid_mask = both_present_mask & (df_copy[payment_date_col] <= df_copy[loss_date_col])

    invalid_count = int(invalid_mask.sum())
    checked_count = int(both_present_mask.sum())
    skipped_count = int(len(df_copy) - checked_count)  # rows missing one or both dates

    # Step 4: mark each row so you can filter/audit later (vectorized, no loop needed)
    df_copy['PAYMENT_VS_LOSS_VALIDATION'] = np.where(
        ~both_present_mask, 'MISSING_DATE',
        np.where(invalid_mask, 'INVALID_PAYMENT_BEFORE_LOSS', 'VALID')
    )

    # Step 5: overall status for the report
    status = "PASS" if invalid_count == 0 else "FAIL"

    return {
        "column": f"{payment_date_col} vs {loss_date_col}",
        "status": status,
        "message": f"{checked_count} rows checked; {skipped_count} skipped (missing date(s)); "
                   f"{invalid_count} invalid (payment date before loss date)",
        "checked_count": checked_count,
        "skipped_count": skipped_count,
        "invalid_count": invalid_count,
        "invalid_rows": df_copy.loc[invalid_mask]   # <- added
    }, df_copy


def VALUATION_VS_LOSS_DATE_VALIDATE(df1, df2, key_col, valuation_date_col, loss_date_col, allow_same_day=True):
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    # Step 0: check if key_col is unique in df2 (duplicates would cause row duplication on merge)
    duplicate_keys_count = int(df2_copy[key_col].duplicated().sum())

    # Step 1: bring valuation date into df1 via merge on the key column
    df_copy = df1_copy.merge(
        df2_copy[[key_col, valuation_date_col]],
        on=key_col,
        how='left'
    )

    # Step 2: make sure both columns are actual datetimes (not strings/objects)
    df_copy[valuation_date_col] = pd.to_datetime(df_copy[valuation_date_col], errors='coerce')
    df_copy[loss_date_col] = pd.to_datetime(df_copy[loss_date_col], errors='coerce')

    # Step 3: only rows where BOTH dates exist can be validated
    both_present_mask = df_copy[valuation_date_col].notnull() & df_copy[loss_date_col].notnull()

    # Step 4: define what "invalid" means -> loss date should be BEFORE (or same as) valuation date
    if allow_same_day:
        invalid_mask = both_present_mask & (df_copy[loss_date_col] > df_copy[valuation_date_col])
    else:
        invalid_mask = both_present_mask & (df_copy[loss_date_col] >= df_copy[valuation_date_col])

    invalid_count = int(invalid_mask.sum())
    checked_count = int(both_present_mask.sum())
    skipped_count = int(len(df_copy) - checked_count)
    unmatched_count = int(df_copy[valuation_date_col].isnull().sum() - df1_copy[loss_date_col].isnull().sum())

    df_copy['VALUATION_VS_LOSS_VALIDATION'] = np.where(
        ~both_present_mask, 'MISSING_DATE',
        np.where(invalid_mask, 'INVALID_LOSS_AFTER_VALUATION', 'VALID')
    )

    status = "PASS" if invalid_count == 0 else "FAIL"

    return {
        "column": f"{loss_date_col} vs {valuation_date_col}",
        "status": status,
        "message": f"{checked_count} rows checked; {skipped_count} skipped (missing date(s)); "
                    f"{invalid_count} invalid (loss date after valuation date); "
                    f"{unmatched_count} rows unmatched in merge with df2; "
                    f"{duplicate_keys_count} duplicate keys found in df2",
        "checked_count": checked_count,
        "skipped_count": skipped_count,
        "invalid_count": invalid_count,
        "unmatched_count": unmatched_count,
        "duplicate_keys_count": duplicate_keys_count,   # <- clean key name, computed early
        "invalid_rows": df_copy.loc[invalid_mask]
    }, df_copy