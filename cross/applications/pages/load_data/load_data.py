import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

from cross.applications.components import next_button


class LoadDataPage:
    def show_page(self):
        st.title("Load data")
        st.write("Load your data or select a toy dataset.")

        # Selector
        option = st.selectbox(
            "Select an option:",
            ("Load CSV", "Toy data: Iris", "Toy data: Wine", "Toy data: Breast Cancer"),
        )

        if option == "Load CSV":
            uploaded_file = st.file_uploader("Select a CSV file", type="csv")

            if uploaded_file:
                st.session_state["data"] = pd.read_csv(uploaded_file)
                st.write(f"### Loaded data from {uploaded_file.name}")

        else:
            st.session_state["data"] = self._load_toy_dataset(option)
            st.write(f"### {option} Dataset")

        enable_next_button = False

        # Display data
        if "data" in st.session_state:
            enable_next_button = True
            st.write(st.session_state["data"].head())

        # Next button
        next_button(disabled=not enable_next_button)

    def _load_toy_dataset(self, name):
        if name == "Toy data: Iris":
            data = load_iris()

        elif name == "Toy data: Wine":
            data = load_wine()

        elif name == "Toy data: Breast Cancer":
            data = load_breast_cancer()

        else:
            return None

        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df
