import pandas as pd


def convert_columns(df, cast_options):
    for column, dtype_to_cast in cast_options.items():
        if dtype_to_cast == "bool":
            df[column] = df[column].astype(bool)

        elif dtype_to_cast == "category":
            df[column] = df[column].astype(str)

        elif dtype_to_cast == "datetime":
            df[column] = pd.to_datetime(df[column])

        elif dtype_to_cast == "number":
            df[column] = pd.to_numeric(df[column], errors="coerce")

        elif dtype_to_cast == "timedelta":
            df[column] = pd.to_timedelta(df[column])

    return df
