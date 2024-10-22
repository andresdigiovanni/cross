import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import is_data_loaded
from cross.applications.styles import plot_remove_borders
from cross.transformations.clean_data import MissingValuesHandler
from cross.transformations.utils.dtypes import categorical_columns, numerical_columns

from .missing_values import MissingValuesBase


class MissingValuesPage(MissingValuesBase):
    def show_page(self):
        st.title("Missing Values Handling")
        st.write(
            "Handle missing values in your DataFrame. "
            "Available options include doing nothing, dropping rows with missing values, "
            "filling missing values with the mean, median, mode, zero, interpolate, etc."
        )

        if not is_data_loaded():
            return

        df, cat_columns, num_columns, valid_columns, missing_values, target_column = (
            self._initialize_data()
        )

        handling_options, n_neighbors = {}, {}

        self._display_missing_values_handling(
            df,
            valid_columns,
            cat_columns,
            num_columns,
            missing_values,
            handling_options,
            n_neighbors,
        )

        st.markdown("""---""")

        self._apply_missing_values_handling(df, handling_options, n_neighbors)

    def _initialize_data(self):
        df = st.session_state["data"]

        cat_columns = categorical_columns(df)
        num_columns = numerical_columns(df)
        missing_values = df.isnull().sum()

        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        columns = [x for x in df.columns if x != target_column]
        valid_columns = [x for x in columns if x in cat_columns + num_columns]

        return (
            df,
            cat_columns,
            num_columns,
            valid_columns,
            missing_values,
            target_column,
        )

    def _display_missing_values_handling(
        self,
        df,
        valid_columns,
        cat_columns,
        num_columns,
        missing_values,
        handling_options,
        n_neighbors,
    ):
        for column in valid_columns:
            st.markdown("""---""")
            col1, col2 = st.columns([2, 1])

            actions = self._get_available_actions(column, cat_columns)

            with col1:
                self._display_column_info(df, column, cat_columns, missing_values)
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
                self._display_column_distribution(df, column, num_columns)

    def _get_available_actions(self, column, cat_columns):
        if column in cat_columns:
            return self.ACTIONS_ALL | self.ACTIONS_CAT

        return self.ACTIONS_ALL | self.ACTIONS_NUM

    def _display_column_info(self, df, column, cat_columns, missing_values):
        st.subheader(column)
        st.write(f"Missing values: {missing_values[column]}")

        if column in cat_columns:
            num_categories = df[column].nunique()
            st.write(f"Number of categories: {num_categories}")

    def _display_column_distribution(self, df, column, num_columns):
        fig, ax = plt.subplots(figsize=(4, 2))

        if column in num_columns:
            sns.histplot(df[column], kde=True, ax=ax, color="#FF4C4B")

        else:
            df[column].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")

        plot_remove_borders(ax)
        st.pyplot(fig)

    def _apply_missing_values_handling(self, df, handling_options, n_neighbors):
        if st.button("Add step"):
            try:
                handling_options_mapped = {
                    col: self.ACTIONS[action]
                    for col, action in handling_options.items()
                    if self.ACTIONS[action] != "none"
                }
                missing_values_handler = MissingValuesHandler(
                    handling_options_mapped, n_neighbors
                )
                transformed_df = missing_values_handler.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = missing_values_handler.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "MissingValuesHandler", "params": params})
                st.session_state["steps"] = steps

                st.success("Missing values handled successfully!")

            except Exception as e:
                st.error(f"Error handling missing values: {e}")
