import streamlit as st
from streamlit_option_menu import option_menu

from cross.applications.pages.clean_data import (
    ColumnSelectionPage,
    MissingValuesPage,
    OutliersHandlingPage,
    RemoveDuplicatesPage,
    TargetSelectionPage,
)
from cross.applications.pages.feature_engineering import (
    CategoricalEncodingPage,
    MathematicalOperationsPage,
    NumericalBinningPage,
)
from cross.applications.pages.load_data import ColumnCastingPage, LoadDataPage
from cross.applications.pages.preprocessing import (
    NonLinearTransformationPage,
    NormalizationPage,
    QuantileTransformationsPage,
    ScaleTransformationsPage,
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
                "Numerical binning",
                "Mathematical operations",
            ],
            "icons": [
                "alphabet",
                "bucket",
                "plus-slash-minus",
            ],
            "pages": [
                CategoricalEncodingPage(),
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
        "icons": pages_icons,
        "pages": {i: k for i, k in enumerate(pages)},
        "page_to_index": {k: i for i, k in enumerate(pages_names)},
        "index_to_page": {i: k for i, k in enumerate(pages_names)},
    }


def navigation_on_change(key):
    selection = st.session_state[key]

    navigation_pages = get_navigation_pages()
    st.session_state["page_index"] = navigation_pages["page_to_index"][selection]


def main():
    st.set_page_config(page_title="CROSS", page_icon="assets/icon.png", layout="wide")

    # Navigation
    if "page_index" not in st.session_state:
        st.session_state["page_index"] = 0

    manual_select = st.session_state["page_index"]
    navigation_pages = get_navigation_pages()

    if navigation_pages["index_to_page"][manual_select] == "---":
        manual_select += 1

    with st.sidebar:
        _, col2, _ = st.columns([0.3, 1, 0.3])
        with col2:
            st.image("assets/logo.png")

        option_menu(
            menu_title=None,
            options=navigation_pages["pages_names"],
            icons=navigation_pages["icons"],
            menu_icon="cast",
            on_change=navigation_on_change,
            key="sidebar_menu",
            manual_select=manual_select,
        )

    # Show page
    if st.session_state["page_index"] in navigation_pages["pages"]:
        navigation_pages["pages"][st.session_state["page_index"]].show_page()


if __name__ == "__main__":
    main()
