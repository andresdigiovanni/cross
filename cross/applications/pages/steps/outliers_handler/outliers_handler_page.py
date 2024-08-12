import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import is_data_loaded, rain_cloud_plot
from cross.core.clean_data.outliers_handler import OutliersHandler
from cross.core.utils.dtypes import numerical_columns

from .outliers_handler import OutliersHandlingBase


class OutliersHandlingPage(OutliersHandlingBase):
    def show_page(self, name):
        st.title("Outliers Handling")
        st.write(
            "Handle outliers in your DataFrame. "
            "Available options include removing outliers, "
            "capping outliers to a threshold, and replacing outliers with the median."
        )

        if not is_data_loaded():
            return

        df = st.session_state["data"]
        num_columns = numerical_columns(df)

        handling_options = {}
        thresholds = {}

        for column in num_columns:
            st.markdown("""---""")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(column)
                selected_action = st.selectbox(
                    f"Action for {column}", self.actions.keys(), key=f"{column}_action"
                )
                selected_action = self.actions[selected_action]
                selected_method = ""

                if selected_action != "none":
                    selected_method = st.selectbox(
                        f"Detection method for {column}",
                        self.detection_methods.keys(),
                        key=f"{column}_method",
                    )
                    selected_method = self.detection_methods[selected_method]

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
                if selected_action != "none":
                    fig, ax = plt.subplots(figsize=(4, 2))
                    selected_action, selected_method = handling_options[column]
                    threshold = thresholds[column]

                    if selected_method == "iqr":
                        sns.boxplot(x=df[column], ax=ax, color="#FF4C4B")

                        q1 = df[column].quantile(0.25)
                        q3 = df[column].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr

                        ax.axvline(lower_bound, color="r", linestyle="--")
                        ax.axvline(upper_bound, color="r", linestyle="--")

                        ax.set_ylabel("Density")
                        ax.set_xlabel(column)

                        # Remove borders
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.spines["left"].set_visible(False)
                        ax.spines["bottom"].set_visible(False)

                    else:
                        sns.histplot(
                            df[column].dropna(), kde=True, ax=ax, color="#FF4C4B"
                        )

                        mean = df[column].mean()
                        std = df[column].std()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std

                        ax.axvline(lower_bound, color="r", linestyle="--")
                        ax.axvline(upper_bound, color="r", linestyle="--")

                        ax.set_xlabel(column)

                        # Remove borders
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.spines["left"].set_visible(False)
                        ax.spines["bottom"].set_visible(False)

                # By default show rain cloud
                else:
                    fig = rain_cloud_plot(df, column)

                st.pyplot(fig)

        st.markdown("""---""")

        # Apply button
        if st.button("Add step"):
            try:
                outliers_handler = OutliersHandler(handling_options, thresholds)
                transformed_df = outliers_handler.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = outliers_handler.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": name, "params": params})
                st.session_state["steps"] = steps

                st.success("Outliers handled successfully!")

            except Exception as e:
                st.error("Error handling outliers: {}".format(e))