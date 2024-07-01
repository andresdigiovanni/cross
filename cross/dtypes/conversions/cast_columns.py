import pandas as pd


def cast_columns(df, cast_options):
    df_transformed = df.copy()

    for column, dtype_to_cast in cast_options.items():
        if dtype_to_cast == "bool":
            df_transformed[column] = df_transformed[column].astype(bool)

        elif dtype_to_cast == "category":
            df_transformed[column] = df_transformed[column].astype(str)

        elif dtype_to_cast == "datetime":
            df_transformed[column] = pd.to_datetime(df_transformed[column])

        elif dtype_to_cast == "number":
            df_transformed[column] = pd.to_numeric(
                df_transformed[column], errors="coerce"
            )

        elif dtype_to_cast == "timedelta":
            df_transformed[column] = pd.to_timedelta(df_transformed[column])

    return df_transformed
