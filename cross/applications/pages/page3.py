import streamlit as st


def next_on_click():
    st.session_state["page_index"] = 0

def show_page():
    st.title("Page 3")
    st.write("Modify column data types.")


    # Next button
    st.button("Next", on_click=next_on_click, type="primary")
