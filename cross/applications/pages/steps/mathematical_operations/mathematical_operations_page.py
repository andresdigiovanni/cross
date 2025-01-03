import streamlit as st

from cross.applications.components import is_data_loaded
from cross.transformations import MathematicalOperations
from cross.transformations.utils.dtypes import numerical_columns

from .mathematical_operations import MathematicalOperationsBase


class MathematicalOperationsPage(MathematicalOperationsBase):
    def show_page(self):
        st.title("Mathematical Operations")
        st.write("Select the mathematical operation for each pair of columns.")

        if not is_data_loaded():
            return

        df, num_columns = self._initialize_data()

        # Initialize operation options if not present
        if "operations_options" not in st.session_state:
            st.session_state.operations_options = [
                (num_columns[0], num_columns[1], "add")
            ]

        self._display_operations(num_columns, df)
        st.button("Add another operation", on_click=self._add_operation)
        st.markdown("""---""")

        self._apply_operations(df)

    def _initialize_data(self):
        config = st.session_state.get("config", {})
        target_column = config.get("target_column", None)

        df = st.session_state["data"]
        num_columns = [col for col in numerical_columns(df) if col != target_column]

        return df, num_columns

    def _add_operation(self):
        num_columns = [col for col in numerical_columns(st.session_state["data"])]
        st.session_state.operations_options.append(
            (num_columns[0], num_columns[1], "add")
        )

    def _display_operations(self, num_columns, original_df):
        for i, (col_a, col_b, operation) in enumerate(
            st.session_state.operations_options
        ):
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            col_a, col_b, operation = self._select_operation(
                i, col_a, col_b, operation, num_columns, col1
            )
            st.session_state.operations_options[i] = (
                col_a,
                col_b,
                self.OPERATIONS[operation],
            )

            self._preview_original_data(original_df, col_a, col_b, col2)
            self._preview_transformed_data(original_df, col_a, col_b, operation, col3)

    def _select_operation(self, i, col_a, col_b, operation, num_columns, col):
        with col:
            col_a = st.selectbox(
                f"Select first column for operation {i + 1}",
                num_columns,
                index=num_columns.index(col_a) if col_a in num_columns else 0,
                key=f"col_a_{i}",
            )
            col_b = st.selectbox(
                f"Select second column for operation {i + 1}",
                num_columns,
                index=num_columns.index(col_b) if col_b in num_columns else 0,
                key=f"col_b_{i}",
            )
            operation = st.selectbox(
                f"Select operation {i + 1}",
                self.OPERATIONS.keys(),
                index=list(self.OPERATIONS.values()).index(operation)
                if operation in self.OPERATIONS.values()
                else 0,
                key=f"operation_{i}",
            )
            return col_a, col_b, operation

    def _preview_original_data(self, original_df, col_a, col_b, col):
        with col:
            st.write("Original Data")
            columns_to_select = list(set([col_a, col_b]))
            st.dataframe(original_df[columns_to_select].head())

    def _preview_transformed_data(self, original_df, col_a, col_b, operation, col):
        with col:
            if self.OPERATIONS[operation] != "none":
                math_operations = MathematicalOperations(
                    [(col_a, col_b, self.OPERATIONS[operation])]
                )
                transformed_df = math_operations.fit_transform(original_df)
                new_column = f"{col_a}__{self.OPERATIONS[operation]}__{col_b}"
                st.write("Transformed Data")
                st.dataframe(transformed_df[new_column].head())
            else:
                st.write("No transformation applied")

    def _apply_operations(self, df):
        if st.button("Add step"):
            try:
                # Apply math operations
                math_operations = MathematicalOperations(
                    st.session_state.operations_options
                )
                transformed_df = math_operations.fit_transform(df)

                # Update session state
                st.session_state["data"] = transformed_df

                params = math_operations.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": "MathematicalOperations", "params": params})
                st.session_state["steps"] = steps

                st.success("Mathematical operations applied successfully!")

            except Exception as e:
                st.error(f"Error applying operations: {e}")
