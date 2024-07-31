import streamlit as st


def is_data_loaded():
    if "data" not in st.session_state:
        st.warning("No data loaded. Please load a DataFrame.")
        return False

    return True
