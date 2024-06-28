import pandas as pd


def datetime_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=["datetime64"]).columns.tolist()
