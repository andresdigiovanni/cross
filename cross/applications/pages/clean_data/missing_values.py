import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import next_button
from cross.preprocessing.dtypes import categorical_columns, numerical_columns
from cross.preprocessing.missing_values import MissingValuesHandler


class MissingValuesPage:
    def show_page(self):
        st.title("Missing Values Handling")
        st.write(
            "Handle missing values in your DataFrame. "
            "Available options include doing nothing, dropping rows with missing values, "
            "filling missing values with the mean, median, mode, zero, interpolate, etc."
        )

        if "data" not in st.session_state:
            st.warning("No data loaded. Please load a DataFrame.")
            return

        df = st.session_state["data"]

        cat_columns = categorical_columns(df)
        num_columns = numerical_columns(df)

        # Actions for each column
        actions_all = {
            "Do nothing": "none",
            "Drop": "drop",
            "Fill with 0": "fill_0",
        }
        actions_cat = {
            "Most frequent": "most_frequent",
        }
        actions_num = {
            "Fill with mean": "fill_mean",
            "Fill with median": "fill_median",
            "Fill with mode": "fill_mode",
            "Interpolate": "interpolate",
            "KNN imputation": "fill_knn",
        }

        handling_options = {}
        n_neighbors = {}
        missing_values = df.isnull().sum()

        valid_columns = [x for x in df.columns if x in cat_columns + num_columns]

        for column in valid_columns:
            st.markdown("""---""")
            col1, col2 = st.columns([2, 1])

            if column in cat_columns:
                actions = actions_all | actions_cat
            else:
                actions = actions_all | actions_num

            with col1:
                st.subheader(column)
                st.write(f"Missing values: {missing_values[column]}")

                if column in cat_columns:
                    num_categories = df[column].nunique()
                    st.write(f"Number of categories: {num_categories}")

                handling_options[column] = st.selectbox(
                    f"Action for {column}", actions.keys(), key=column
                )

                if actions[handling_options[column]] == "fill_knn":
                    neighbors = st.slider(
                        f"Select number of neighbors for {column}",
                        min_value=1,
                        max_value=20,
                        value=5,
                        key=f"{column}_neighbors",
                    )
                    n_neighbors[column] = neighbors

            with col2:
                fig, ax = plt.subplots(figsize=(4, 2))
                if column in num_columns:
                    sns.histplot(df[column], kde=True, ax=ax)
                else:
                    df[column].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                    ax.set_ylabel("")
                st.pyplot(fig)

        st.markdown("""---""")

        # Convert button
        if st.button("Apply Actions"):
            try:
                actions = actions_all | actions_cat | actions_num

                handling_options_mapped = {
                    col: actions[action] for col, action in handling_options.items()
                }
                missing_values_handler = MissingValuesHandler(
                    handling_options_mapped, n_neighbors
                )
                df = missing_values_handler.fit_transform(df)
                st.session_state["data"] = df

                config = st.session_state.get("config", {})
                config["missing_values"] = {
                    "handling_options": missing_values_handler.handling_options.copy(),
                    "statistics": missing_values_handler.statistics_.copy(),
                }
                st.session_state["config"] = config

                st.success("Missing values handled successfully!")

            except Exception as e:
                st.error("Error handling missing values: {}".format(e))

        # Next button
        next_button()
