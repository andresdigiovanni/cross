import streamlit as st


def _next_on_click():
    st.session_state["page_index"] = st.session_state["page_index"] + 1


def next_button(disabled=False):
    return st.button(
        "Next step", disabled=disabled, on_click=_next_on_click, type="primary"
    )
