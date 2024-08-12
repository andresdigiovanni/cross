import streamlit as st

from cross.applications.pages.steps import (
    CategoricalEncodingEdit,
    ColumnCastingEdit,
    ColumnSelectionEdit,
    CyclicalFeaturesTransformationEdit,
    DateTimeTransformationEdit,
    MathematicalOperationsEdit,
    MissingValuesEdit,
    NonLinearTransformationEdit,
    NormalizationEdit,
    NumericalBinningEdit,
    OutliersHandlingEdit,
    QuantileTransformationsEdit,
    RemoveDuplicatesEdit,
    ScaleTransformationsEdit,
)


class ModifyStepsPage:
    def show_page(self):
        st.title("Modify steps")
        st.write("---")

        steps = st.session_state.get("steps", [])
        checkboxes = []

        for i, step in enumerate(steps):
            name = step["name"]
            params = step["params"]

            checkbox = st.checkbox("**{} - {}**".format(i + 1, name))
            checkboxes.append(checkbox)

            edit_page = self._get_edit_page(name)
            edit_page.show_component(params)

            st.write("---")

        st.session_state["modify_steps_checkbox"] = checkboxes
        st.button("Remove", on_click=self._remove_steps, type="primary")

    def _get_edit_page(self, name):
        if name == "Categorical encoding":
            return CategoricalEncodingEdit()

        elif name == "Column casting":
            return ColumnCastingEdit()

        elif name == "Column selection":
            return ColumnSelectionEdit()

        elif name == "Cyclical transforms":
            return CyclicalFeaturesTransformationEdit()

        elif name == "Datetime transforms":
            return DateTimeTransformationEdit()

        elif name == "Mathematical operations":
            return MathematicalOperationsEdit()

        elif name == "Missing values":
            return MissingValuesEdit()

        elif name == "Non-linear transforms":
            return NonLinearTransformationEdit()

        elif name == "Normalize":
            return NormalizationEdit()

        elif name == "Numerical binning":
            return NumericalBinningEdit()

        elif name == "Handle outliers":
            return OutliersHandlingEdit()

        elif name == "Quantile transforms":
            return QuantileTransformationsEdit()

        elif name == "Remove duplicates":
            return RemoveDuplicatesEdit()

        elif name == "Scale":
            return ScaleTransformationsEdit()

    def _remove_steps(self):
        steps = st.session_state.get("steps", [])
        checkboxes = st.session_state.get("modify_steps_checkbox", [])

        steps_to_discard = [i for i, checked in enumerate(checkboxes) if checked]
        steps = [step for i, step in enumerate(steps) if i not in steps_to_discard]

        st.session_state["steps"] = steps
