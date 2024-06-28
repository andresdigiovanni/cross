import pages.column_casting
import pages.load_data
import streamlit as st
from streamlit_option_menu import option_menu


def get_navigation_pages():
    pages_names = ["Load data", "Column casting"]
    icons = ["upload", "shuffle"]

    return {
        "pages": pages_names,
        "icons": icons,
        "page_to_index": {k: i for i, k in enumerate(pages_names)},
        "index_to_page": {i: k for i, k in enumerate(pages_names)},
    }


def navigation_on_change(key):
    selection = st.session_state[key]

    navigation_pages = get_navigation_pages()
    st.session_state["page_index"] = navigation_pages["page_to_index"][selection]


def main():
    st.set_page_config(page_title="Cross", layout="wide")

    if "page_index" not in st.session_state:
        st.session_state["page_index"] = 0

    manual_select = st.session_state["page_index"]
    navigation_pages = get_navigation_pages()

    # Navigation
    with st.sidebar:
        option_menu(
            menu_title="",
            options=navigation_pages["pages"],
            icons=navigation_pages["icons"],
            menu_icon="cast",
            on_change=navigation_on_change,
            key="sidebar_menu",
            manual_select=manual_select,
        )

    # Show page
    if st.session_state["page_index"] == 0:
        pages.load_data.show_page()

    elif st.session_state["page_index"] == 1:
        pages.column_casting.show_page()


if __name__ == "__main__":
    main()
