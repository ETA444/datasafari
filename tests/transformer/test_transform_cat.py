import pytest
import pandas as pd
import numpy as np
from datasafari.transformer.transform_cat import transform_cat


@pytest.fixture
def sample_data():
    """ Provides sample data for tests. """
    return pd.DataFrame({
        'Category': np.random.choice(['A', 'B', 'C'], size=100),
        'Target': np.random.choice([1, 0], size=100)
    })


# TESTING ERROR-HANDLING #
def test_invalid_df_type():
    """ Test that non-DataFrame input raises a TypeError. """
    with pytest.raises(TypeError, match="The 'df' parameter must be a pandas DataFrame."):
        transform_cat("not_a_dataframe", ['Category'], method='uniform_simple')


def test_invalid_categorical_variables_type(sample_data):
    """ Test that non-list categorical_variables input raises a TypeError. """
    with pytest.raises(TypeError, match="The 'categorical_variables' parameter must be a list of column names."):
        transform_cat(sample_data, 'Category', method='uniform_simple')


def test_empty_dataframe():
    """ Test handling of empty DataFrame. """
    with pytest.raises(ValueError, match="The input DataFrame is empty."):
        transform_cat(pd.DataFrame(), ['Category'], method='uniform_simple')


def test_nonexistent_categorical_variable(sample_data):
    """ Test handling of non-existent categorical variable. """
    with pytest.raises(ValueError, match="The following variables were not found in the DataFrame:"):
        transform_cat(sample_data, ['Nonexistent'], method='uniform_simple')


def test_invalid_method(sample_data):
    """ Test handling of invalid method input. """
    with pytest.raises(ValueError, match="Invalid method 'not_a_valid_method'"):
        transform_cat(sample_data, ['Category'], method='not_a_valid_method')


def test_missing_abbreviation_map(sample_data):
    """ Test that missing abbreviation_map raises a ValueError when needed. """
    with pytest.raises(ValueError, match="The 'abbreviation_map' parameter must be provided when using the 'uniform_mapping' method."):
        transform_cat(sample_data, ['Category'], method='uniform_mapping')


def test_missing_ordinal_map(sample_data):
    """ Test that missing ordinal_map raises a ValueError when needed. """
    with pytest.raises(ValueError, match="The 'ordinal_map' parameter must be provided when using the 'encode_ordinal' method."):
        transform_cat(sample_data, ['Category'], method='encode_ordinal')


def test_invalid_abbreviation_map_type(sample_data):
    """ Test that non-dictionary abbreviation_map input raises a TypeError. """
    with pytest.raises(TypeError, match="The 'abbreviation_map' parameter must be a dictionary if provided."):
        transform_cat(sample_data, ['Category'], method='uniform_mapping', abbreviation_map='not_a_dict')


def test_invalid_ordinal_map_type(sample_data):
    """ Test that non-dictionary ordinal_map input raises a TypeError. """
    with pytest.raises(TypeError, match="The 'ordinal_map' parameter must be a dictionary if provided."):
        transform_cat(sample_data, ['Category'], method='encode_ordinal', ordinal_map='not_a_dict')


def test_missing_target_variable_for_target_encoding(sample_data):
    """ Test handling of missing target_variable for target encoding. """
    with pytest.raises(ValueError, match="The 'target_variable' parameter must be provided when using the 'encode_target' method."):
        transform_cat(sample_data, ['Category'], method='encode_target')


def test_nonexistent_target_variable(sample_data):
    """ Test handling of non-existent target_variable. """
    with pytest.raises(ValueError, match="The target variable 'Nonexistent' was not found in the DataFrame."):
        transform_cat(sample_data, ['Category'], method='encode_target', target_variable='Nonexistent')


def test_target_variable_not_string(sample_data):
    """ Test that non-string target_variable raises a TypeError. """
    with pytest.raises(TypeError, match="The 'target_variable' parameter must be a string if provided."):
        transform_cat(sample_data, ['Category'], method='encode_target', target_variable=123)


def test_categorical_variables_empty_list(sample_data):
    """ Test handling of empty categorical_variables list. """
    with pytest.raises(ValueError, match="The 'categorical_variables' list must contain at least one column name."):
        transform_cat(sample_data, [], method='uniform_simple')
