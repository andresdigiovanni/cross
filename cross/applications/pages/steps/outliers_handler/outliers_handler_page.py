import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from cross.applications.components import is_data_loaded
from cross.applications.styles import plot_remove_borders
from cross.core.clean_data import OutliersHandler
from cross.core.utils.dtypes import numerical_columns

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
        num_columns = numerical_columns(df)
        num_columns = [x for x in num_columns if x != target_column]

        handling_options = {}
        thresholds = {}
        lof_params = {}
        iforest_params = {}

        rows_affected = {}

        for column in num_columns:
            st.markdown("""---""")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(column)
                selected_action = st.selectbox(
                    f"Action for {column}", self.actions.keys(), key=f"{column}_action"
                )
                selected_action = self.actions[selected_action]

                selected_method = st.selectbox(
                    f"Detection method for {column}",
                    self.detection_methods.keys(),
                    key=f"{column}_method",
                )
                selected_method = self.detection_methods[selected_method]

                if selected_method == "lof":
                    n_neighbors = st.slider(
                        "Select number of neighbors",
                        min_value=5,
                        max_value=50,
                        value=20,
                        step=1,
                        key=f"{column}_lof_neighbors",
                    )
                    lof_params[column] = {"n_neighbors": n_neighbors}

                elif selected_method == "iforest":
                    contamination = st.slider(
                        "Select contamination level",
                        min_value=0.01,
                        max_value=0.5,
                        value=0.1,
                        step=0.01,
                        key=f"{column}_iforest_contamination",
                    )
                    iforest_params[column] = {"contamination": contamination}
                else:
                    threshold = st.slider(
                        "Select threshold",
                        min_value=1.0,
                        max_value=3.0 if selected_method == "iqr" else 5.0,
                        value=1.5 if selected_method == "iqr" else 3.0,
                        step=0.1,
                        key=f"{column}_threshold",
                    )
                    thresholds[column] = threshold

                handling_options[column] = (selected_action, selected_method)

            with col2:
                fig, ax = plt.subplots(figsize=(4, 2))

                if selected_method == "iqr":
                    sns.boxplot(x=df[column], ax=ax, color="#FF4C4B")

                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - thresholds[column] * iqr
                    upper_bound = q3 + thresholds[column] * iqr

                    ax.axvline(lower_bound, color="r", linestyle="--")
                    ax.axvline(upper_bound, color="r", linestyle="--")

                    rows_affected[column] = df[
                        (df[column] < lower_bound) | (df[column] > upper_bound)
                    ].shape[0]

                    ax.set_ylabel("Density")
                    ax.set_xlabel(column)

                    plot_remove_borders(ax)

                elif selected_method == "zscore":
                    sns.histplot(df[column].dropna(), kde=True, ax=ax, color="#FF4C4B")

                    mean = df[column].mean()
                    std = df[column].std()
                    lower_bound = mean - thresholds[column] * std
                    upper_bound = mean + thresholds[column] * std

                    ax.axvline(lower_bound, color="r", linestyle="--")
                    ax.axvline(upper_bound, color="r", linestyle="--")

                    rows_affected[column] = df[
                        (df[column] < lower_bound) | (df[column] > upper_bound)
                    ].shape[0]

                    ax.set_xlabel(column)
                    plot_remove_borders(ax)

                elif selected_method == "lof":
                    lof = LocalOutlierFactor(
                        n_neighbors=lof_params[column]["n_neighbors"]
                    )
                    y_pred = lof.fit_predict(df[[column]].dropna())
                    is_outlier = y_pred == -1

                    rows_affected[column] = is_outlier.sum()

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

                elif selected_method == "iforest":
                    iforest = IsolationForest(
                        contamination=iforest_params[column]["contamination"]
                    )
                    y_pred = iforest.fit_predict(df[[column]].dropna())
                    is_outlier = y_pred == -1

                    rows_affected[column] = is_outlier.sum()

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

                st.pyplot(fig)

            st.write(f"Rows affected in {column}: {rows_affected[column]}")

        st.markdown("""---""")

        # Apply button
        if st.button("Add step"):
            try:
                outliers_handler = OutliersHandler(
                    handling_options, thresholds, lof_params, iforest_params
                )
                transformed_df = outliers_handler.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = outliers_handler.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "OutliersHandler", "params": params})
                st.session_state["steps"] = steps

                st.success("Outliers handled successfully!")

            except Exception as e:
                st.error("Error handling outliers: {}".format(e))
