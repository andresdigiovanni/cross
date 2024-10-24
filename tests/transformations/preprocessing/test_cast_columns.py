import pandas as pd

from cross.transformations.preprocessing.cast_columns import CastColumns


class TestCastColumns:

    # Cast columns to specified data types using cast_options
    def test_cast_columns_to_specified_data_types(self):
        data = {
            'bool_col': [1, 0, 1],
            'category_col': [1, 2, 3],
            'datetime_col': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'number_col': ['1', '2', '3'],
            'timedelta_col': ['1 days', '2 days', '3 days']
        }
        df = pd.DataFrame(data)
    
        cast_options = {
            'bool_col': 'bool',
            'category_col': 'category',
            'datetime_col': 'datetime',
            'number_col': 'number',
            'timedelta_col': 'timedelta'
        }
    
        transformer = CastColumns(cast_options=cast_options)
        transformed_df = transformer.fit_transform(df)
    
        assert transformed_df['bool_col'].dtype == bool
        assert transformed_df['category_col'].dtype.name == 'object'
        assert pd.api.types.is_datetime64_any_dtype(transformed_df['datetime_col'])
        assert pd.api.types.is_numeric_dtype(transformed_df['number_col'])
        assert pd.api.types.is_timedelta64_dtype(transformed_df['timedelta_col'])
