import pandas as pd


def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort columns of the DataFrame based on a specific order.
    """
    col_order = ["path", "section", "is_normal", "is_target"]
    sorted_cols = [col for col in col_order if col in df.columns]
    sorted_cols += sorted(list(set(df.columns) - set(sorted_cols)))
    return df[sorted_cols].copy()
