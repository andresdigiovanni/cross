import streamlit as st

from cross.applications.pages.navigation_pages import navigation_pages


class ModifyStepsPage:
    def show_page(self):
        st.title("Modify Steps")
        st.write("---")

        # Retrieve the steps and pages from session state and navigation pages
        steps = st.session_state.get("steps", [])
        pages = navigation_pages()

        # Display each step with checkbox and editing functionality
        self._display_steps_with_editing(steps, pages)

        # Button to remove selected steps
        st.button("Remove", on_click=self._remove_selected_steps, type="primary")

    def _display_steps_with_editing(self, steps, pages):
        checkboxes = []

        for index, step in enumerate(steps):
            # Extract step details
            step_name, step_params = step["name"], step["params"]
            display_name = pages["key_to_name"][step_name]

            # Checkbox for selecting the step
            checkbox = st.checkbox(f"**{index + 1} - {display_name}**")
            checkboxes.append(checkbox)

            # Show the editing component for the step
            self._show_edit_component(pages, display_name, step_params)
            st.write("---")

        # Store checkboxes in session state
        st.session_state["modify_steps_checkbox"] = checkboxes

    def _show_edit_component(self, pages, display_name, params):
        edit_page = self._get_edit_page(pages, display_name)
        if edit_page:
            edit_page.show_component(params)

    def _get_edit_page(self, pages, name):
        page_index = pages["name_to_index"][name]
        page_edit = pages["index_to_edit"][page_index]

        return page_edit

    def _remove_selected_steps(self):
        steps = st.session_state.get("steps", [])
        checkboxes = st.session_state.get("modify_steps_checkbox", [])

        # Filter out steps that are checked
        st.session_state["steps"] = [
            step for i, step in enumerate(steps) if not checkboxes[i]
        ]
