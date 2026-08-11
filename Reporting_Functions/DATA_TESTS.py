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
    """
    Checks for missing As-At Dates using check_missing().
    Fills missing values with the quarter-end date of the Payment Date.
    Re-validates the column and returns a cleaning report.
    """

    df_copy = df.copy()

    df_copy[as_at_col] = pd.to_datetime(df_copy[as_at_col], errors='coerce')

    # Step 1: detect missing values using the existing validator
    before_check = check_missing(df, as_at_col)

    # If column doesn't exist or nothing is missing, nothing to fix
    if before_check["status"] in ["PASS", "ERROR"]:
        return {
            "column": as_at_col,
            "status": before_check["status"],
            "message": before_check["message"],
            "rows_fixed": 0
        }

    # Step 2: identify the missing rows
    missing_mask = df[as_at_col].isnull()

    # Step 3: fix - substitute with the quarter-end date of the payment date
    quarter_end_dates = df.loc[missing_mask, payment_col].dt.to_period('Q').dt.end_time
    df.loc[missing_mask, as_at_col] = quarter_end_dates

    # Step 4: re-validate to confirm the fix worked
    after_check = check_missing(df, as_at_col)

    # Step 5: return a full report, not just a silent fix
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
    }, df_copy


def VALUATION_VS_LOSS_DATE_VALIDATE(df1, df2, key_col, valuation_date_col, loss_date_col, allow_same_day=True):
    """
    Validates that loss dates occur on or before valuation dates and tags each row with validation results.
    df1                 : main dataframe containing loss_date_col and key_col
    df2                 : second dataframe (from another sheet) containing valuation_date_col and key_col
    key_col             : column name used to join df1 and df2 (must exist in both, same name)
    valuation_date_col  : name of the valuation date column (in df2)
    loss_date_col       : name of the loss date column (in df1)
    Returns: dict report with keys: column, status, message, checked_count, skipped_count,
    invalid_count, unmatched_count, df, invalid_rows
    """

    df1_copy = df1.copy()
    df2_copy = df2.copy()

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
    skipped_count = int(len(df_copy) - checked_count)  # rows missing one or both dates (incl. failed merges)
    unmatched_count = int(df_copy[valuation_date_col].isnull().sum() - df1_copy[loss_date_col].isnull().sum())

    # Step 5: mark each row so you can filter/audit later
    df_copy['VALUATION_VS_LOSS_VALIDATION'] = np.where(
        ~both_present_mask, 'MISSING_DATE',
        np.where(invalid_mask, 'INVALID_LOSS_AFTER_VALUATION', 'VALID')
    )

    # Step 6: overall status for the report
    status = "PASS" if invalid_count == 0 else "FAIL"

    return {
        "column": f"{loss_date_col} vs {valuation_date_col}",
        "status": status,
        "message": f"{checked_count} rows checked; {skipped_count} skipped (missing date(s)); "
                    f"{invalid_count} invalid (loss date after valuation date); "
                    f"{unmatched_count} rows unmatched in merge with df2",
        "checked_count": checked_count,
        "skipped_count": skipped_count,
        "invalid_count": invalid_count,
        "unmatched_count": unmatched_count,
    }, df_copy


def MASTER_DATA_QUALITY_REPORT(
    df1,
    df2=None,
    category_col=None,
    loss_date_col='LA_LOSS_DATE',
    as_at_col='LA_AS_AT_DATE',
    payment_date_col='LA_PAYMENT_DATE',
    status_col='LA_STATUS',
    condition_value='PAID',
    key_col=None,
    valuation_date_col=None,
    allow_same_day=True
):
    """
    Runs the full validation/cleaning pipeline in the correct order and
    returns a single consolidated report + the final cleaned dataframe.

    Order matters:
      1. check_missing        -> raw missing checks (category, loss date) BEFORE any cleaning
      2. AS_AT_DATE_CLEAN     -> fills missing as-at dates using payment date
      3. PAYMENT_DATE_CLEAN   -> wipes payment dates for non-PAID rows
      4. PAYMENT_VS_LOSS_DATE_VALIDATE   -> uses the now-cleaned payment date
      5. VALUATION_VS_LOSS_DATE_VALIDATE -> only runs if df2/key_col/valuation_date_col given
    """

    report = []
    df_working = df1.copy()

    # ---- 1. Raw missing-value checks (run BEFORE cleaning, on original data) ----
    if category_col:
        cat_check = check_missing(df_working, category_col)
        report.append({
            "step": "check_missing (category)",
            "column": cat_check["column"],
            "status": cat_check["status"],
            "detail": cat_check["message"]
        })

    loss_check = check_missing(df_working, loss_date_col)
    report.append({
        "step": "check_missing (loss date)",
        "column": loss_check["column"],
        "status": loss_check["status"],
        "detail": loss_check["message"]
    })

    # ---- 2. Fill missing As-At dates ----
    df_working[payment_date_col] = pd.to_datetime(df_working[payment_date_col], errors='coerce')
    as_at_result = AS_AT_DATE_CLEAN(df_working, as_at_col, payment_date_col)
    report.append({
        "step": "AS_AT_DATE_CLEAN",
        "column": as_at_result["column"],
        "status": as_at_result["status"],
        "detail": as_at_result["message"]
    })

    # ---- 3. Wipe payment dates for non-PAID rows ----
    df_working, wiped_count, missing_paid_dates = PAYMENT_DATE_CLEAN(
        df_working, payment_date_col, status_col, condition_value
    )
    report.append({
        "step": "PAYMENT_DATE_CLEAN",
        "column": payment_date_col,
        "status": "INFO",
        "detail": f"{wiped_count} dates wiped (status != '{condition_value}'); "
                  f"{missing_paid_dates} PAID rows still missing a payment date"
    })

    # ---- 4. Payment date must be after loss date ----
    df_working, invalid_pay_count, invalid_pay_rows = PAYMENT_VS_LOSS_DATE_VALIDATE(
        df_working, payment_date_col, loss_date_col, allow_same_day
    )
    report.append({
        "step": "PAYMENT_VS_LOSS_DATE_VALIDATE",
        "column": f"{payment_date_col} vs {loss_date_col}",
        "status": "PASS" if invalid_pay_count == 0 else "FAIL",
        "detail": f"{invalid_pay_count} rows have payment date before loss date"
    })

    # ---- 5. Loss date must be before valuation date (optional, needs df2) ----
    invalid_val_rows = None
    if df2 is not None and key_col and valuation_date_col:
        df_working, invalid_val_count, invalid_val_rows = VALUATION_VS_LOSS_DATE_VALIDATE(
            df_working, df2, key_col, valuation_date_col, loss_date_col, allow_same_day
        )
        report.append({
            "step": "VALUATION_VS_LOSS_DATE_VALIDATE",
            "column": f"{loss_date_col} vs {valuation_date_col}",
            "status": "PASS" if invalid_val_count == 0 else "FAIL",
            "detail": f"{invalid_val_count} rows have loss date after valuation date"
        })
    else:
        report.append({
            "step": "VALUATION_VS_LOSS_DATE_VALIDATE",
            "column": "-",
            "status": "SKIPPED",
            "detail": "df2 / key_col / valuation_date_col not provided"
        })

    # ---- Build final summary dataframe ----
    report_df = pd.DataFrame(report)

    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)
    print(report_df.to_string(index=False))
    print("=" * 70)

    return df_working, report_df, {
        "invalid_payment_rows": invalid_pay_rows,
        "invalid_valuation_rows": invalid_val_rows
    }