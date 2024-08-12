import streamlit as st

from .outliers_handler import OutliersHandlingBase


class OutliersHandlingEdit(OutliersHandlingBase):
    def show_component(self, params):
        handling_options = params["handling_options"]
        thresholds = params["thresholds"]
        reverse_actions = {v: k for k, v in self.actions.items()}
        reverse_detection = {v: k for k, v in self.detection_methods.items()}

        markdown = [
            "| Column | Action | Method | Threshold |",
            "| --- | --- | --- | --- |",
        ]

        for column, (action, method) in handling_options.items():
            if action == "none":
                continue

            markdown.append(
                "| {} | {} | {} | {} |".format(
                    column,
                    reverse_actions[action],
                    reverse_detection[method],
                    thresholds[column],
                )
            )

        st.markdown("\n".join(markdown))
