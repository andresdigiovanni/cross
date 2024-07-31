import streamlit as st

from cross.applications.components.check_is_data_loaded import is_data_loaded
from cross.core.feature_engineering.mathematical_operations import (
    MathematicalOperations,
)
from cross.core.utils.dtypes import numerical_columns


class MathematicalOperationsPage:
    def show_page(self, name):
        st.title("Mathematical Operations")
        st.write("Select the mathematical operation for each pair of columns.")

        if not is_data_loaded():
            return

        df = st.session_state["data"]
        original_df = df.copy()

        num_columns = numerical_columns(df)

        # Available operations
        operations = {
            "Do nothing": "none",
            "Addition": "add",
            "Subtraction": "subtract",
            "Multiplication": "multiply",
            "Division": "divide",
            "Modulus": "modulus",
            "Power": "power",
            "Hypotenuse": "hypotenuse",
            "Mean": "mean",
        }

        if "operations_options" not in st.session_state:
            st.session_state.operations_options = [
                (num_columns[0], num_columns[1], "add")
            ]

        def add_operation():
            st.session_state.operations_options.append(
                (num_columns[0], num_columns[1], "add")
            )

        for i, (col_a, col_b, operation) in enumerate(
            st.session_state.operations_options
        ):
            st.markdown("""---""")
            col1, col2, col3 = st.columns(3)

            with col1:
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
                    operations.keys(),
                    index=list(operations.values()).index(operation)
                    if operation in operations.values()
                    else 0,
                    key=f"operation_{i}",
                )
                st.session_state.operations_options[i] = (
                    col_a,
                    col_b,
                    operations[operation],
                )

            with col2:
                st.write("Original Data")
                columns_to_select = list(set([col_a, col_b]))
                st.dataframe(original_df[columns_to_select].head())

            with col3:
                if operations[operation] != "none":
                    math_operations = MathematicalOperations(
                        [(col_a, col_b, operations[operation])]
                    )
                    transformed_df = math_operations.fit_transform(original_df)

                    new_column = f"{col_a}__{operations[operation]}__{col_b}"

                    st.write("Transformed Data")
                    st.dataframe(transformed_df[new_column].head())
                else:
                    st.write("No transformation applied")

        st.button("Add another operation", on_click=add_operation)
        st.markdown("""---""")

        # Apply button
        if st.button("Add step"):
            try:
                math_operations = MathematicalOperations(
                    st.session_state.operations_options
                )
                transformed_df = math_operations.fit_transform(df)
                st.session_state["data"] = transformed_df

                params = math_operations.get_params()
                steps = st.session_state.get("steps", [])
                steps.append({"name": name, "params": params})
                st.session_state["steps"] = steps

                st.success("Mathematical operations applied successfully!")

            except Exception as e:
                st.error("Error applying operations: {}".format(e))
