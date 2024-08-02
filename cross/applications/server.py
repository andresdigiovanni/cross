import pickle

import streamlit as st
from streamlit_option_menu import option_menu

from cross.applications.pages import (
    CategoricalEncodingPage,
    ColumnCastingPage,
    ColumnSelectionPage,
    CyclicalFeaturesTransformationPage,
    DateTimeTransformationPage,
    LoadDataPage,
    MathematicalOperationsPage,
    MissingValuesPage,
    NonLinearTransformationPage,
    NormalizationPage,
    NumericalBinningPage,
    OutliersHandlingPage,
    QuantileTransformationsPage,
    RemoveDuplicatesPage,
    ScaleTransformationsPage,
    TargetSelectionPage,
)


def get_navigation_pages():
    pages_hierarchy = [
        {
            "name": "Load data",
            "pages_names": ["Load data", "Column casting"],
            "icons": ["upload", "shuffle"],
            "pages": [LoadDataPage(), ColumnCastingPage()],
        },
        {
            "name": "Clean data",
            "pages_names": [
                "Column selection",
                "Target selection",
                "Remove duplicates",
                "Handle outliers",
                "Missing values",
            ],
            "icons": [
                "list-check",
                "bullseye",
                "copy",
                "distribute-horizontal",
                "question-octagon",
            ],
            "pages": [
                ColumnSelectionPage(),
                TargetSelectionPage(),
                RemoveDuplicatesPage(),
                OutliersHandlingPage(),
                MissingValuesPage(),
            ],
        },
        {
            "name": "Preprocessing",
            "pages_names": [
                "Non-linear transforms",
                "Quantile transforms",
                "Scale",
                "Normalize",
            ],
            "icons": [
                "bar-chart-steps",
                "bezier2",
                "arrows-angle-expand",
                "bounding-box",
            ],
            "pages": [
                NonLinearTransformationPage(),
                QuantileTransformationsPage(),
                ScaleTransformationsPage(),
                NormalizationPage(),
            ],
        },
        {
            "name": "Feature engineering",
            "pages_names": [
                "Categorical encoding",
                "Datetime transforms",
                "Cyclical transforms",
                "Numerical binning",
                "Mathematical operations",
            ],
            "icons": [
                "alphabet",
                "calendar-date",
                "arrow-clockwise",
                "bucket",
                "plus-slash-minus",
            ],
            "pages": [
                CategoricalEncodingPage(),
                DateTimeTransformationPage(),
                CyclicalFeaturesTransformationPage(),
                NumericalBinningPage(),
                MathematicalOperationsPage(),
            ],
        },
    ]

    pages = []
    pages_names = []
    pages_icons = []

    for i, page_hierarchy in enumerate(pages_hierarchy):
        if i > 0:
            pages.append(None)
            pages_names.append("---")
            pages_icons.append(None)

        pages.extend(page_hierarchy["pages"])
        pages_names.extend(page_hierarchy["pages_names"])
        pages_icons.extend(page_hierarchy["icons"])

    return {
        "pages_names": pages_names,
        "pages": {i: k for i, k in enumerate(pages)},
        "page_to_index": {k: i for i, k in enumerate(pages_names)},
        "index_to_page": {i: k for i, k in enumerate(pages_names)},
        "icons": pages_icons,
        "index_to_icon": {i: k for i, k in enumerate(pages_icons)},
    }


def navigation_on_change(key):
    selection = st.session_state[key]

    navigation_pages = get_navigation_pages()
    st.session_state["page_index"] = navigation_pages["page_to_index"][selection]


def save_config():
    config = st.session_state.get("config", {})
    with open("config.pkl", "wb") as f:
        pickle.dump(config, f)
    st.success("Configuration saved to config.pkl")


def main():
    st.set_page_config(page_title="CROSS", page_icon="assets/icon.png", layout="wide")

    # Navigation
    if "page_index" not in st.session_state:
        st.session_state["page_index"] = 0

    manual_select = st.session_state["page_index"]
    navigation_pages = get_navigation_pages()

    if navigation_pages["index_to_page"][manual_select] == "---":
        manual_select += 1

    # Sidebar
    with st.sidebar:
        _, col2, _ = st.columns([0.3, 1, 0.3])
        with col2:
            st.image("assets/logo.png")

        option_menu(
            menu_title=None,
            options=navigation_pages["pages_names"],
            icons=navigation_pages["icons"],
            on_change=navigation_on_change,
            key="sidebar_menu",
            manual_select=manual_select,
        )

    col1, col2 = st.columns([3, 1], gap="medium")

    # List of operations
    with col1:
        page_index = st.session_state["page_index"]
        page_name = navigation_pages["pages_names"][page_index]
        navigation_pages["pages"][page_index].show_page(page_name)

    # Selected operations
    with col2:
        st.subheader("Steps")
        steps = st.session_state.get("steps", [])

        if len(steps) == 0:
            st.write("No selected operations")

        else:
            for i, step in enumerate(steps):
                name = step["name"]
                st.write(f"{i + 1}. {name}")

            st.write("---")

            # Add buttons
            col1_buttons, col2_buttons = st.columns([1, 1])

            with col1_buttons:
                if st.button("Modify"):
                    save_config()  # FIXME

            with col2_buttons:
                if st.button("Save", type="primary"):
                    save_config()


if __name__ == "__main__":
    main()
