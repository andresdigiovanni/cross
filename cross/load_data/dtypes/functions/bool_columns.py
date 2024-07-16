import pandas as pd


def bool_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=["bool"]).columns.tolist()
