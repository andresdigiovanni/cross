import pandas as pd


def categorical_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=["object", "category"]).columns.tolist()
