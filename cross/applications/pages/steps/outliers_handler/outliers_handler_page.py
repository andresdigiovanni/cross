import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from cross.applications.components import is_data_loaded
from cross.applications.styles import plot_remove_borders
from cross.transformations import OutliersHandler
from cross.transformations.utils.dtypes import numerical_columns

from .outliers_handler import OutliersHandlingBase


class OutliersHandlingPage(OutliersHandlingBase):
    def show_page(self):
        st.title("Outliers Handling")
        st.write(
            "Handle outliers in your DataFrame. "
            "Available options include removing outliers, "
            "capping outliers to a threshold, and replacing outliers with the median."
        )

        if not is_data_loaded():
            return

        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        df = st.session_state["data"]
        num_columns = [col for col in numerical_columns(df) if col != target_column]

        handling_options, thresholds, lof_params, iforest_params, rows_affected = (
            {},
            {},
            {},
            {},
            {},
        )

        for column in num_columns:
            st.markdown("---")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(column)
                selected_method = self._configure_column_method(column)

                if selected_method == "lof":
                    lof_params[column] = self._get_lof_params(column)

                elif selected_method == "iforest":
                    iforest_params[column] = self._get_iforest_params(column)

                else:
                    thresholds[column] = self._get_threshold(column, selected_method)

                selected_action = self._configure_column_action(column, selected_method)

                handling_options[column] = (selected_action, selected_method)

            with col2:
                rows_affected[column] = self._plot_column_outliers(
                    df, column, selected_method, thresholds, lof_params, iforest_params
                )

            st.write(f"Rows affected in {column}: {rows_affected[column]}")

        st.markdown("---")

        if st.button("Add step"):
            self._apply_outliers_handler(
                df, handling_options, thresholds, lof_params, iforest_params
            )

    def _configure_column_method(self, column):
        selected_method = st.selectbox(
            f"Detection method for {column}",
            self.DETECTION_METHODS.keys(),
            key=f"{column}_method",
        )
        selected_method = self.DETECTION_METHODS[selected_method]

        return selected_method

    def _configure_column_action(self, column, selected_method):
        actions = [
            x
            for x in self.ACTIONS.keys()
            if not (selected_method in ["lof", "iforest"] and x == "Cap to threshold")
        ]
        selected_action = st.selectbox(
            f"Action for {column}", actions, key=f"{column}_action"
        )
        selected_action = self.ACTIONS[selected_action]

        return selected_action

    def _get_lof_params(self, column):
        n_neighbors = st.slider(
            f"Select number of neighbors for {column}",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
            key=f"{column}_lof_neighbors",
        )
        return {"n_neighbors": n_neighbors}

    def _get_iforest_params(self, column):
        contamination = st.slider(
            f"Select contamination level for {column}",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            key=f"{column}_iforest_contamination",
        )
        return {"contamination": contamination}

    def _get_threshold(self, column, selected_method):
        max_value = 3.0 if selected_method == "iqr" else 5.0
        default_value = 1.5 if selected_method == "iqr" else 3.0

        return st.slider(
            f"Select threshold for {column}",
            min_value=1.0,
            max_value=max_value,
            value=default_value,
            step=0.1,
            key=f"{column}_threshold",
        )

    def _plot_column_outliers(
        self, df, column, selected_method, thresholds, lof_params, iforest_params
    ):
        fig, ax = plt.subplots(figsize=(4, 2))

        if selected_method == "iqr":
            n_rows_affected = self._plot_iqr(df, column, thresholds[column], ax)

        elif selected_method == "zscore":
            n_rows_affected = self._plot_zscore(df, column, thresholds[column], ax)

        elif selected_method == "lof":
            n_rows_affected = self._plot_lof(df, column, lof_params[column], ax)

        elif selected_method == "iforest":
            n_rows_affected = self._plot_iforest(df, column, iforest_params[column], ax)

        st.pyplot(fig)
        return n_rows_affected

    def _plot_iqr(self, df, column, threshold, ax):
        sns.boxplot(x=df[column], ax=ax, color="#FF4C4B")

        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - threshold * iqr, q3 + threshold * iqr

        ax.axvline(lower_bound, color="r", linestyle="--")
        ax.axvline(upper_bound, color="r", linestyle="--")
        ax.set_ylabel("Density")
        ax.set_xlabel(column)
        plot_remove_borders(ax)

        return df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]

    def _plot_zscore(self, df, column, threshold, ax):
        sns.histplot(df[column].dropna(), kde=True, ax=ax, color="#FF4C4B")

        mean, std = df[column].mean(), df[column].std()
        lower_bound, upper_bound = mean - threshold * std, mean + threshold * std

        ax.axvline(lower_bound, color="r", linestyle="--")
        ax.axvline(upper_bound, color="r", linestyle="--")
        ax.set_xlabel(column)
        plot_remove_borders(ax)

        return df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]

    def _plot_lof(self, df, column, lof_params, ax):
        lof = LocalOutlierFactor(n_neighbors=lof_params["n_neighbors"])
        is_outlier = lof.fit_predict(df[[column]].dropna()) == -1

        sns.scatterplot(
            x=df.index,
            y=df[column],
            hue=is_outlier,
            palette={True: "red", False: "blue"},
            ax=ax,
        )
        ax.set_xlabel("Index")
        ax.set_ylabel(column)
        ax.get_legend().remove()
        plot_remove_borders(ax)

        return is_outlier.sum()

    def _plot_iforest(self, df, column, iforest_params, ax):
        iforest = IsolationForest(contamination=iforest_params["contamination"])
        is_outlier = iforest.fit_predict(df[[column]].dropna()) == -1

        sns.scatterplot(
            x=df.index,
            y=df[column],
            hue=is_outlier,
            palette={True: "red", False: "blue"},
            ax=ax,
        )
        ax.set_xlabel("Index")
        ax.set_ylabel(column)
        ax.get_legend().remove()
        plot_remove_borders(ax)

        return is_outlier.sum()

    def _apply_outliers_handler(
        self, df, handling_options, thresholds, lof_params, iforest_params
    ):
        try:
            outliers_handler = OutliersHandler(
                {k: v for k, v in handling_options.items() if v[0] != "none"},
                thresholds,
                lof_params,
                iforest_params,
            )
            transformed_df = outliers_handler.fit_transform(df)
            st.session_state["data"] = transformed_df

            params = outliers_handler.get_params()
            steps = st.session_state.get("steps", [])
            steps.append({"name": "OutliersHandler", "params": params})
            st.session_state["steps"] = steps

            st.success("Outliers handled successfully!")

        except Exception as e:
            st.error(f"Error handling outliers: {e}")
