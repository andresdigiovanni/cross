import streamlit as st


def next_on_click():
    st.session_state["page_index"] = st.session_state["page_index"] + 1
