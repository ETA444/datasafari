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


# TESTING FUNCTIONALITY #

def test_transform_cat_uniform_simple(sample_data):
    """ Test uniform simple transformation for categorical data. """
    transformed_df, transformed_columns = transform_cat(sample_data, ['Category'], method='uniform_simple')
    assert 'Category' in transformed_df.columns
    assert transformed_df['Category'].isna().sum() == 0  # Check no NaN values
    assert transformed_df['Category'].str.islower().all()  # Check all values are lowercase


def test_transform_cat_uniform_smart(sample_data):
    """ Test uniform smart transformation with textual similarity clustering. """
    transformed_df, transformed_columns = transform_cat(sample_data, ['Category'], method='uniform_smart')
    assert 'Category' in transformed_df.columns
    # Verify transformation maintains data integrity
    unique_original = len(sample_data['Category'].unique())
    unique_transformed = len(transformed_df['Category'].unique())
    assert unique_transformed <= unique_original  # Should be less or equal due to clustering


def test_transform_cat_uniform_mapping(sample_data):
    """ Test manual mapping of categories. """
    abbreviation_map = {'Category': {'A': 'Alpha', 'B': 'Beta', 'C': 'Charlie'}}
    transformed_df, transformed_columns = transform_cat(sample_data, ['Category'], method='uniform_mapping', abbreviation_map=abbreviation_map)
    expected_values = set(['Alpha', 'Beta', 'Charlie'])
    assert set(transformed_df['Category'].unique()).issubset(expected_values)


def test_transform_cat_encode_onehot(sample_data):
    """ Test one-hot encoding of categories. """
    transformed_df, transformed_columns = transform_cat(sample_data, ['Category'], method='encode_onehot')
    # Check for the presence of new columns
    expected_columns = ['Category_A', 'Category_B', 'Category_C']
    for col in expected_columns:
        assert col in transformed_columns.columns


def test_transform_cat_encode_ordinal(sample_data):
    """ Test ordinal encoding of categories. """
    ordinal_map = {'Category': ['A', 'B', 'C']}
    transformed_df, transformed_columns = transform_cat(sample_data, ['Category'], method='encode_ordinal', ordinal_map=ordinal_map)
    assert transformed_df['Category'].dtype == 'int8'  # Check if encoded to integers
    assert set(transformed_df['Category'].unique()).issubset({0, 1, 2})


def test_transform_cat_encode_freq(sample_data):
    """ Test frequency encoding of categories. """
    transformed_df, transformed_columns = transform_cat(sample_data, ['Category'], method='encode_freq')
    # Verify that frequency counts replace category names
    assert transformed_df['Category'].dtype == 'int64'
    assert transformed_df['Category'].min() > 0  # Frequency should be positive integers


def test_transform_cat_encode_target(sample_data):
    """ Test target encoding based on the mean of the target variable. """
    transformed_df, transformed_columns = transform_cat(sample_data, ['Category'], method='encode_target', target_variable='Target')
    # Verify that target means replace category names and are floats
    assert transformed_df['Category'].dtype == float


def test_transform_cat_encode_binary(sample_data):
    """ Test binary encoding of categories. """
    transformed_df, transformed_columns = transform_cat(sample_data, ['Category'], method='encode_binary')
    # Check if the binary encoding added the expected columns
    assert 'Category_0' in transformed_columns.columns
    assert 'Category_1' in transformed_columns.columns
