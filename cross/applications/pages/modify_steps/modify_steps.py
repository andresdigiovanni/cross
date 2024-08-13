import streamlit as st

from cross.applications.pages.navigation_pages import navigation_pages


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
        pages = navigation_pages()

        page_index = pages["page_to_index"][name]
        page_edit = pages["index_to_edit"][page_index]

        return page_edit

    def _remove_steps(self):
        steps = st.session_state.get("steps", [])
        checkboxes = st.session_state.get("modify_steps_checkbox", [])

        steps_to_discard = [i for i, checked in enumerate(checkboxes) if checked]
        steps = [step for i, step in enumerate(steps) if i not in steps_to_discard]

        st.session_state["steps"] = steps
