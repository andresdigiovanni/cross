import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from cross.applications.components import next_button
from cross.clean_data.outliers_handler import OutliersHandler
from cross.load_data.dtypes import numerical_columns


class OutliersHandlingPage:
    def show_page(self):
        st.title("Outliers Handling")
        st.write(
            "Handle outliers in your DataFrame. "
            "Available options include removing outliers, "
            "capping outliers to a threshold, and replacing outliers with the median."
        )

        if "data" not in st.session_state:
            st.warning("No data loaded. Please load a DataFrame.")
            return

        df = st.session_state["data"]
        num_columns = numerical_columns(df)

        # Actions for each column
        actions = {
            "Do nothing": "none",
            "Remove": "remove",
            "Cap to threshold": "cap",
            "Replace with median": "median",
        }
        detection_methods = {
            "IQR": "iqr",
            "Z-score": "zscore",
        }

        handling_options = {}
        thresholds = {}

        for column in num_columns:
            st.markdown("""---""")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(column)
                selected_action = st.selectbox(
                    f"Action for {column}", actions.keys(), key=f"{column}_action"
                )
                selected_action = actions[selected_action]
                selected_method = ""

                if selected_action != "none":
                    selected_method = st.selectbox(
                        f"Detection method for {column}",
                        detection_methods.keys(),
                        key=f"{column}_method",
                    )
                    selected_method = detection_methods[selected_method]

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
                        sns.boxplot(x=df[column], ax=ax)
                        q1 = df[column].quantile(0.25)
                        q3 = df[column].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        ax.axvline(lower_bound, color="r", linestyle="--")
                        ax.axvline(upper_bound, color="r", linestyle="--")
                        ax.set_title(f"IQR for {column}")
                        ax.set_xlabel(column)

                    else:
                        sns.histplot(df[column].dropna(), kde=True, ax=ax)
                        mean = df[column].mean()
                        std = df[column].std()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                        ax.axvline(lower_bound, color="r", linestyle="--")
                        ax.axvline(upper_bound, color="r", linestyle="--")
                        ax.set_title(f"Z-score Distribution for {column}")
                        ax.set_ylabel("Density")
                        ax.set_xlabel(column)

                    st.pyplot(fig)

        st.markdown("""---""")

        # Apply button
        if st.button("Apply Actions"):
            try:
                outliers_handler = OutliersHandler(handling_options, thresholds)
                df = outliers_handler.fit_transform(df)
                st.session_state["data"] = df

                config = st.session_state.get("config", {})
                config["outliers_handling"] = {
                    "handling_options": outliers_handler.handling_options.copy(),
                    "statistics": outliers_handler.statistics_.copy(),
                    "bounds": outliers_handler.bounds.copy(),
                }
                st.session_state["config"] = config

                st.success("Outliers handled successfully!")

            except Exception as e:
                st.error("Error handling outliers: {}".format(e))

        # Next button
        next_button()
