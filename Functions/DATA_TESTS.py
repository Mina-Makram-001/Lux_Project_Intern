import pandas as pd


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
    }


def PAYMENT_DATE_CLEAN():
    pass


def PAYMENT_VS_LOSS_DATE_VALIDATE():
    pass


def VALUATION_VS_LOSS_DATE_VALIDATE():
    pass