import numpy as np
import pandas as pd


def numerical_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=np.number).columns.tolist()
