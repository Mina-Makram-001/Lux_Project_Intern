import pandas as pd

def load_excel(file_path: str) -> pd.DataFrame:
    """
    Load an Excel file into a pandas DataFrame and standardize 
    all date columns to 'YYYY-MM-DD' string format.

    Parameters:
    file_path (str): The path to the Excel file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data with formatted dates.
    """
    try:
        df = pd.read_excel(file_path)
        
        # Iterate over all columns and convert date-like columns to YYYY-MM-DD
        for col in df.columns:
            # Check if column is already datetime type or contains date strings
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d')
            elif df[col].dtype == 'object':
                # Attempt to parse object/string columns containing dates
                parsed = pd.to_datetime(df[col], errors='coerce', format='mixed')
                # If more than 50% of non-null values successfully parse as dates, apply formatting
                if parsed.notna().sum() > 0.5 * df[col].notna().sum():
                    df[col] = parsed.dt.strftime('%Y-%m-%d')

        return df

    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    