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
    ScaleTransformationsEdit,
    ScaleTransformationsPage,
    TargetSelectionPage,
)


def navigation_pages():
    pages_hierarchy = [
        # Common Data Preparation Tasks
        [
            {
                "key": "LoadData",
                "name": "Load data",
                "icon": "upload",
                "page": LoadDataPage(),
                "edit": None,
            },
            {
                "key": "TargetSelection",
                "name": "Target selection",
                "icon": "bullseye",
                "page": TargetSelectionPage(),
                "edit": None,
            },
            {
                "key": "ColumnSelection",
                "name": "Column selection",
                "icon": "list-check",
                "page": ColumnSelectionPage(),
                "edit": ColumnSelectionEdit(),
            },
            {
                "key": "CastColumns",
                "name": "Column casting",
                "icon": "shuffle",
                "page": ColumnCastingPage(),
                "edit": ColumnCastingEdit(),
            },
        ],
        # Data Cleaning
        [
            {
                "key": "MissingValuesHandler",
                "name": "Missing values",
                "icon": "question-octagon",
                "page": MissingValuesPage(),
                "edit": MissingValuesEdit(),
            },
            {
                "key": "OutliersHandler",
                "name": "Handle outliers",
                "icon": "distribute-horizontal",
                "page": OutliersHandlingPage(),
                "edit": OutliersHandlingEdit(),
            },
        ],
        # Data Transforms - Numerical
        [
            {
                "key": "NonLinearTransformation",
                "name": "Non-linear transforms",
                "icon": "bar-chart-steps",
                "page": NonLinearTransformationPage(),
                "edit": NonLinearTransformationEdit(),
            },
            {
                "key": "QuantileTransformation",
                "name": "Quantile transforms",
                "icon": "bezier2",
                "page": QuantileTransformationsPage(),
                "edit": QuantileTransformationsEdit(),
            },
            {
                "key": "ScaleTransformation",
                "name": "Scale",
                "icon": "arrows-angle-expand",
                "page": ScaleTransformationsPage(),
                "edit": ScaleTransformationsEdit(),
            },
            {
                "key": "Normalization",
                "name": "Normalize",
                "icon": "bounding-box",
                "page": NormalizationPage(),
                "edit": NormalizationEdit(),
            },
        ],
        # Data Transforms - Categorical
        [
            {
                "key": "CategoricalEncoding",
                "name": "Categorical encoding",
                "icon": "alphabet",
                "page": CategoricalEncodingPage(),
                "edit": CategoricalEncodingEdit(),
            },
            {
                "key": "DateTimeTransformer",
                "name": "Datetime transforms",
                "icon": "calendar-date",
                "page": DateTimeTransformationPage(),
                "edit": DateTimeTransformationEdit(),
            },
            {
                "key": "CyclicalFeaturesTransformer",
                "name": "Cyclical transforms",
                "icon": "arrow-clockwise",
                "page": CyclicalFeaturesTransformationPage(),
                "edit": CyclicalFeaturesTransformationEdit(),
            },
        ],
        # Feature engineering
        [
            {
                "key": "NumericalBinning",
                "name": "Numerical binning",
                "icon": "bucket",
                "page": NumericalBinningPage(),
                "edit": NumericalBinningEdit(),
            },
            {
                "key": "MathematicalOperations",
                "name": "Mathematical operations",
                "icon": "plus-slash-minus",
                "page": MathematicalOperationsPage(),
                "edit": MathematicalOperationsEdit(),
            },
        ],
    ]

    pages_keys = []
    pages_names = []
    pages_icons = []
    pages_show = []
    pages_edit = []

    for i, subpages in enumerate(pages_hierarchy):
        if i > 0:
            pages_keys.append("---")
            pages_names.append("---")
            pages_icons.append(None)
            pages_show.append(None)
            pages_edit.append(None)

        pages_keys.extend([page["key"] for page in subpages])
        pages_names.extend([page["name"] for page in subpages])
        pages_icons.extend([page["icon"] for page in subpages])
        pages_show.extend([page["page"] for page in subpages])
        pages_edit.extend([page["edit"] for page in subpages])

    return {
        "pages_names": pages_names,
        "pages_icons": pages_icons,
        "name_to_index": {k: i for i, k in enumerate(pages_names)},
        "key_to_name": {k: n for k, n in zip(pages_keys, pages_names)},
        "index_to_name": {i: k for i, k in enumerate(pages_names)},
        "index_to_show": {i: k for i, k in enumerate(pages_show)},
        "index_to_edit": {i: k for i, k in enumerate(pages_edit)},
    }
