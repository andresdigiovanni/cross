import pandas as pd


def timedelta_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=["timedelta64"]).columns.tolist()
