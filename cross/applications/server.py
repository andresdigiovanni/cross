import pickle

import streamlit as st
from streamlit_option_menu import option_menu

from cross.applications.pages.modify_steps import ModifyStepsPage
from cross.applications.pages.navigation_pages import navigation_pages
from cross.applications.styles import css


def navigation_on_change(key):
    selection = st.session_state[key]

    pages = navigation_pages()
    st.session_state["page_index"] = pages["page_to_index"][selection]
    st.session_state["is_show_modify_page"] = False


def save_config():
    config = st.session_state.get("steps", {})
    with open("cross_transformations.pkl", "wb") as f:
        pickle.dump(config, f)

    st.success("Configuration saved to cross_transformations.pkl")


def modify_steps():
    st.session_state["page_index"] = -1
    st.session_state["is_show_modify_page"] = True


def main():
    st.set_page_config(page_title="CROSS", page_icon="assets/icon.png", layout="wide")
    st.logo("assets/logo.png")

    # Style
    css()

    # Navigation
    if "page_index" not in st.session_state:
        st.session_state["page_index"] = 0

    manual_select = st.session_state["page_index"]
    pages = navigation_pages()

    if pages["index_to_page"].get(manual_select, "") == "---":
        manual_select += 1

    if "is_show_modify_page" not in st.session_state:
        st.session_state["is_show_modify_page"] = False

    # Sidebar
    with st.sidebar:
        option_menu(
            menu_title=None,
            options=pages["pages_names"],
            icons=pages["pages_icons"],
            on_change=navigation_on_change,
            key="sidebar_menu",
            manual_select=manual_select,
        )

    col1, col2 = st.columns([3, 1], gap="medium")

    # List of operations
    with col1:
        if st.session_state["is_show_modify_page"]:
            ModifyStepsPage().show_page()

        else:
            page_index = st.session_state["page_index"]
            pages["pages_show"][page_index].show_page()

    # Selected operations
    with col2:
        st.subheader("Steps")
        steps = st.session_state.get("steps", [])

        if len(steps) == 0:
            st.write("No selected operations")

        else:
            for i, step in enumerate(steps):
                name = step["name"]
                st.write(f"{i + 1} - {name}")

            st.write("---")

            # Add buttons
            col1_buttons, col2_buttons = st.columns([1, 1])

            with col1_buttons:
                st.button("Modify", on_click=modify_steps)

            with col2_buttons:
                if st.button("Save", type="primary"):
                    save_config()


if __name__ == "__main__":
    main()
