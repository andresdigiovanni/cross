import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine


def load_toy_dataset(name):
    if name == "Iris":
        data = load_iris()

    elif name == "Wine":
        data = load_wine()

    elif name == "Breast Cancer":
        data = load_breast_cancer()

    else:
        return None

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def next_on_click():
    st.session_state["page_index"] = st.session_state["page_index"] + 1


def show_page():
    st.title("Load data")
    st.write("Load your data or select a toy dataset")

    st.session_state["df"] = None
    enable_button = False

    # Selector
    option = st.selectbox(
        "Select an option:",
        ("Load CSV", "Toy data: Iris", "Toy data: Wine", "Toy data: Breast Cancer"),
    )

    if option == "Load CSV":
        uploaded_file = st.file_uploader("Select a CSV file", type="csv")

        if uploaded_file:
            st.session_state["df"] = pd.read_csv(uploaded_file)
            st.write(f"### Loaded data from {uploaded_file.name}")

    else:
        st.session_state["df"] = load_toy_dataset(option)
        st.write(f"### {option} Dataset")

    # Display data
    if st.session_state["df"] is not None:
        enable_button = True
        st.write(st.session_state["df"].head())

    # Next button
    st.button("Next", disabled=not enable_button, on_click=next_on_click)
