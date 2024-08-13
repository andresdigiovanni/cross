from cross.applications.pages.steps import (
    CategoricalEncodingEdit,
    CategoricalEncodingPage,
    ColumnCastingEdit,
    ColumnCastingPage,
    ColumnSelectionEdit,
    ColumnSelectionPage,
    CyclicalFeaturesTransformationEdit,
    CyclicalFeaturesTransformationPage,
    DateTimeTransformationEdit,
    DateTimeTransformationPage,
    LoadDataPage,
    MathematicalOperationsEdit,
    MathematicalOperationsPage,
    MissingValuesEdit,
    MissingValuesPage,
    NonLinearTransformationEdit,
    NonLinearTransformationPage,
    NormalizationEdit,
    NormalizationPage,
    NumericalBinningEdit,
    NumericalBinningPage,
    OutliersHandlingEdit,
    OutliersHandlingPage,
    QuantileTransformationsEdit,
    QuantileTransformationsPage,
    RemoveDuplicatesEdit,
    RemoveDuplicatesPage,
    ScaleTransformationsEdit,
    ScaleTransformationsPage,
    TargetSelectionPage,
)


def navigation_pages():
    pages_hierarchy = [
        # Common Data Preparation Tasks
        [
            {
                "name": "Load data",
                "icon": "upload",
                "page": LoadDataPage(),
                "edit": None,
            },
            {
                "name": "Column selection",
                "icon": "list-check",
                "page": ColumnSelectionPage(),
                "edit": ColumnSelectionEdit(),
            },
            {
                "name": "Target selection",
                "icon": "bullseye",
                "page": TargetSelectionPage(),
                "edit": None,
            },
            {
                "name": "Column casting",
                "icon": "shuffle",
                "page": ColumnCastingPage(),
                "edit": ColumnCastingEdit(),
            },
        ],
        # Data Cleaning
        [
            {
                "name": "Remove duplicates",
                "icon": "copy",
                "page": RemoveDuplicatesPage(),
                "edit": RemoveDuplicatesEdit(),
            },
            {
                "name": "Missing values",
                "icon": "question-octagon",
                "page": MissingValuesPage(),
                "edit": MissingValuesEdit(),
            },
            {
                "name": "Handle outliers",
                "icon": "distribute-horizontal",
                "page": OutliersHandlingPage(),
                "edit": OutliersHandlingEdit(),
            },
        ],
        # Data Transforms - Numerical
        [
            {
                "name": "Non-linear transforms",
                "icon": "bar-chart-steps",
                "page": NonLinearTransformationPage(),
                "edit": NonLinearTransformationEdit(),
            },
            {
                "name": "Quantile transforms",
                "icon": "bezier2",
                "page": QuantileTransformationsPage(),
                "edit": QuantileTransformationsEdit(),
            },
            {
                "name": "Scale",
                "icon": "arrows-angle-expand",
                "page": ScaleTransformationsPage(),
                "edit": ScaleTransformationsEdit(),
            },
            {
                "name": "Normalize",
                "icon": "bounding-box",
                "page": NormalizationPage(),
                "edit": NormalizationEdit(),
            },
        ],
        # Data Transforms - Categorical
        [
            {
                "name": "Categorical encoding",
                "icon": "alphabet",
                "page": CategoricalEncodingPage(),
                "edit": CategoricalEncodingEdit(),
            },
            {
                "name": "Datetime transforms",
                "icon": "calendar-date",
                "page": DateTimeTransformationPage(),
                "edit": DateTimeTransformationEdit(),
            },
            {
                "name": "Cyclical transforms",
                "icon": "arrow-clockwise",
                "page": CyclicalFeaturesTransformationPage(),
                "edit": CyclicalFeaturesTransformationEdit(),
            },
        ],
        # Feature engineering
        [
            {
                "name": "Numerical binning",
                "icon": "bucket",
                "page": NumericalBinningPage(),
                "edit": NumericalBinningEdit(),
            },
            {
                "name": "Mathematical operations",
                "icon": "plus-slash-minus",
                "page": MathematicalOperationsPage(),
                "edit": MathematicalOperationsEdit(),
            },
        ],
    ]

    pages_show = []
    pages_edit = []
    pages_names = []
    pages_icons = []

    for i, subpages in enumerate(pages_hierarchy):
        if i > 0:
            pages_show.append(None)
            pages_edit.append(None)
            pages_names.append("---")
            pages_icons.append(None)

        pages_show.extend([page["page"] for page in subpages])
        pages_edit.extend([page["edit"] for page in subpages])
        pages_names.extend([page["name"] for page in subpages])
        pages_icons.extend([page["icon"] for page in subpages])

    return {
        "pages_names": pages_names,
        "pages_icons": pages_icons,
        "pages_show": {i: k for i, k in enumerate(pages_show)},
        "pages_edit": {i: k for i, k in enumerate(pages_edit)},
        "page_to_index": {k: i for i, k in enumerate(pages_names)},
        "index_to_page": {i: k for i, k in enumerate(pages_names)},
        "index_to_icon": {i: k for i, k in enumerate(pages_icons)},
        "index_to_edit": {i: k for i, k in enumerate(pages_edit)},
    }
