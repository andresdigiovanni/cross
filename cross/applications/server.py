import pages.p01_column_casting
import pages.p01_load_data
import pages.p02_column_selection
import pages.p02_data_missing
import pages.p03_normalization
import streamlit as st
from streamlit_option_menu import option_menu


def get_navigation_pages():
    pages_hierarchy = [
        {
            "name": "Load data",
            "pages": ["Load data", "Column casting"],
            "icons": ["upload", "shuffle"],
        },
        {
            "name": "Clean data",
            "pages": ["Column selection", "Missing values"],
            "icons": ["list-check", "question-octagon"],
        },
        {
            "name": "Normalization",
            "pages": ["Normalization"],
            "icons": ["plus-slash-minus"],
        },
    ]

    pages_names = []
    pages_icons = []

    for i, page_hierarchy in enumerate(pages_hierarchy):
        if i > 0:
            pages_names.append("---")
            pages_icons.append(None)

        pages_names.extend(page_hierarchy["pages"])
        pages_icons.extend(page_hierarchy["icons"])

    return {
        "pages": pages_names,
        "icons": pages_icons,
        "page_to_index": {k: i for i, k in enumerate(pages_names)},
        "index_to_page": {i: k for i, k in enumerate(pages_names)},
    }


def navigation_on_change(key):
    selection = st.session_state[key]

    navigation_pages = get_navigation_pages()
    st.session_state["page_index"] = navigation_pages["page_to_index"][selection]


def main():
    st.set_page_config(page_title="Cross", layout="wide")

    # Navigation
    if "page_index" not in st.session_state:
        st.session_state["page_index"] = 0

    manual_select = st.session_state["page_index"]
    navigation_pages = get_navigation_pages()

    if navigation_pages["index_to_page"][manual_select] == "---":
        manual_select += 1

    with st.sidebar:
        option_menu(
            menu_title=None,
            options=navigation_pages["pages"],
            icons=navigation_pages["icons"],
            menu_icon="cast",
            on_change=navigation_on_change,
            key="sidebar_menu",
            manual_select=manual_select,
        )

    # Show page
    if st.session_state["page_index"] == 0:
        pages.p01_load_data.show_page()

    elif st.session_state["page_index"] == 1:
        pages.p01_column_casting.show_page()

    elif st.session_state["page_index"] == 3:
        pages.p02_column_selection.show_page()

    elif st.session_state["page_index"] == 4:
        pages.p02_data_missing.show_page()

    elif st.session_state["page_index"] == 6:
        pages.p03_normalization.show_page()


if __name__ == "__main__":
    main()
