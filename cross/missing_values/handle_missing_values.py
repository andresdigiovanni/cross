def handle_missing_values(df, handling_options):
    df_transformed = df.copy()

    for column, action in handling_options.items():
        if action == "drop":
            df_transformed = df_transformed.dropna(subset=[column])

        elif action == "fill_mean":
            df_transformed[column] = df_transformed[column].fillna(
                df_transformed[column].mean()
            )

        elif action == "fill_median":
            df_transformed[column] = df_transformed[column].fillna(
                df_transformed[column].median()
            )

        elif action == "fill_mode":
            df_transformed[column] = df_transformed[column].fillna(
                df_transformed[column].mode()[0]
            )

        elif action == "fill_0":
            df_transformed[column] = df_transformed[column].fillna(0)

        elif action == "interpolate":
            df_transformed[column] = df_transformed[column].interpolate()

    return df_transformed
