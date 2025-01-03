import numpy as np
import pandas as pd
import pytest

from cross.transformations import CorrelatedSubstringEncoder


class TestCorrelatedSubstringEncoder:
    def test_encode_substrings_into_new_columns(self):
        data = pd.DataFrame({
            'text': ['apple pie', 'banana split', 'cherry tart']
        })
        substrings = {
            'text': ['apple', 'banana', 'cherry']
        }

        encoder = CorrelatedSubstringEncoder(substrings=substrings)
        transformed_data = encoder.fit_transform(data)

        expected_data = pd.DataFrame({
            'text': ['apple pie', 'banana split', 'cherry tart'],
            'text__corr_substring': ['apple', 'banana', 'cherry']
        })

        pd.testing.assert_frame_equal(transformed_data, expected_data)

    def test_handle_empty_input_data(self):
        data = pd.DataFrame(columns=['text'])
        substrings = {'text': ['apple', 'banana', 'cherry']}

        encoder = CorrelatedSubstringEncoder(substrings=substrings)
        transformed_data = encoder.fit_transform(data)

        expected_data = pd.DataFrame(columns=['text', 'text__corr_substring'])

        pd.testing.assert_frame_equal(transformed_data, expected_data)

    def test_no_substring_matches(self):
        data = pd.DataFrame({'text': ['grape', 'orange', 'melon']})
        substrings = {'text': ['apple', 'banana', 'cherry']}
        
        encoder = CorrelatedSubstringEncoder(substrings=substrings)
        transformed_data = encoder.fit_transform(data)

        expected_data = pd.DataFrame({
            'text': ['grape', 'orange', 'melon'],
            'text__corr_substring': [encoder.DEFAULT_VALUE] * 3
        })

        pd.testing.assert_frame_equal(transformed_data, expected_data)


    def test_handle_empty_strings(self):
        data = pd.DataFrame({'text': ['apple pie', '', 'cherry tart']})
        substrings = {'text': ['apple', 'banana', 'cherry']}
        
        encoder = CorrelatedSubstringEncoder(substrings=substrings)
        transformed_data = encoder.fit_transform(data)

        expected_data = pd.DataFrame({
            'text': ['apple pie', '', 'cherry tart'],
            'text__corr_substring': ['apple', encoder.DEFAULT_VALUE, 'cherry']
        })

        pd.testing.assert_frame_equal(transformed_data, expected_data)

    def test_overlapping_substrings(self):
        data = pd.DataFrame({'text': ['apple pie', 'banana split', 'cherry tart']})
        substrings = {'text': ['app', 'apple', 'banana']}
        
        encoder = CorrelatedSubstringEncoder(substrings=substrings)
        transformed_data = encoder.fit_transform(data)

        expected_data = pd.DataFrame({
            'text': ['apple pie', 'banana split', 'cherry tart'],
            'text__corr_substring': ['app', 'banana', encoder.DEFAULT_VALUE]
        })

        pd.testing.assert_frame_equal(transformed_data, expected_data)

    def test_column_not_in_substrings(self):
        data = pd.DataFrame({'text': ['apple pie', 'banana split', 'cherry tart'], 'other_column': [1, 2, 3]})
        substrings = {'text': ['apple', 'banana', 'cherry']}
        
        encoder = CorrelatedSubstringEncoder(substrings=substrings)
        transformed_data = encoder.fit_transform(data)

        expected_data = pd.DataFrame({
            'text': ['apple pie', 'banana split', 'cherry tart'],
            'other_column': [1, 2, 3],
            'text__corr_substring': ['apple', 'banana', 'cherry']
        })

        pd.testing.assert_frame_equal(transformed_data, expected_data)

    def test_empty_substring_list(self):
        data = pd.DataFrame({'text': ['apple pie', 'banana split', 'cherry tart']})
        substrings = {'text': []}
        
        encoder = CorrelatedSubstringEncoder(substrings=substrings)
        transformed_data = encoder.fit_transform(data)

        expected_data = pd.DataFrame({
            'text': ['apple pie', 'banana split', 'cherry tart'],
            'text__corr_substring': [encoder.DEFAULT_VALUE] * 3
        })

        pd.testing.assert_frame_equal(transformed_data, expected_data)
