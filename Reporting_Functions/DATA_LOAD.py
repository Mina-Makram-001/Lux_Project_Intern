import pandas as pd

def load_excel(file_path: str):
    """
    Loads an Excel file into a pandas DataFrame.
    """
    try:
        df = pd.read_excel(file_path)
        return df

    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return pd.DataFrame()