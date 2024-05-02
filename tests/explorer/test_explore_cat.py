import pytest
import pandas as pd
import numpy as np
from datasafari.explorer.explore_cat import explore_cat


@pytest.fixture
def complex_categorical_df():
    return pd.DataFrame({
        'Age': np.random.randint(18, 35, 100),
        'Income': np.random.normal(50000, 15000, 100),
        'Department': np.random.choice(['HR', 'Tech', 'Admin'], 100)
    })


# TESTING ERROR-HANDLING #
def test_empty_dataframe():
    """
    Verify the function handles an empty DataFrame by raising a ValueError.
    """
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="The input DataFrame is empty."):
        explore_cat(df, ['Category'])


def test_non_dataframe_input():
    """
    Ensure that passing a non-DataFrame as input raises a TypeError.
    """
    with pytest.raises(TypeError, match="The df parameter must be a pandas DataFrame."):
        explore_cat("not a dataframe", ['Department'])


def test_non_existent_column(complex_categorical_df):
    """
    Check that providing non-existent column names raises a ValueError.
    """
    with pytest.raises(ValueError):
        explore_cat(complex_categorical_df, ['NonExistentColumn'])


def test_non_categorical_column(complex_categorical_df):
    """
    Ensure that non-categorical columns raise a ValueError when included in categorical_variables.
    """
    with pytest.raises(ValueError, match="The 'categorical_variables' list must contain only names of categorical variables."):
        explore_cat(complex_categorical_df, ['Age', 'Department'])


def test_empty_categorical_variables_list(complex_categorical_df):
    """
    Confirm that providing an empty list for categorical_variables raises a ValueError.
    """
    with pytest.raises(ValueError, match="The 'categorical_variables' list must contain at least one column name."):
        explore_cat(complex_categorical_df, [])


def test_non_list_categorical_variables(complex_categorical_df):
    """
    Validate that non-list categorical_variables input raises a TypeError.
    """
    with pytest.raises(TypeError, match="The categorical_variables parameter must be a list of variable names."):
        explore_cat(complex_categorical_df, 'Department')  # Incorrect type


def test_non_string_in_categorical_variables(complex_categorical_df):
    """
    Ensure that non-string entries in categorical_variables list raise a TypeError.
    """
    with pytest.raises(TypeError, match="All items in the categorical_variables list must be strings representing column names."):
        explore_cat(complex_categorical_df, [123, 'Department'])  # Non-string element


def test_invalid_method(complex_categorical_df):
    """
    Test that providing an invalid method raises the appropriate ValueError.
    """
    with pytest.raises(ValueError):
        explore_cat(complex_categorical_df, ['Department'], method='invalid_method')


def test_invalid_output_option(complex_categorical_df):
    """
    Confirm that providing an invalid output option raises a ValueError.
    """
    with pytest.raises(ValueError, match="Invalid output method. Choose 'print' or 'return'."):
        explore_cat(complex_categorical_df, ['Department'], output='invalid_output')


# TESTING FUNCTIONALITY #
def test_entropy_method(complex_categorical_df):
    """
    Verify that entropy calculations are performed and return expected string outputs.
    """
    result = explore_cat(complex_categorical_df, ['Department'], method='entropy', output='return')
    assert 'Entropy' in result
    assert '1.585' in result


def test_counts_percentage_output(complex_categorical_df):
    """
    Test the correct output format and values for counts and percentage method.
    """
    result = explore_cat(complex_categorical_df, ['Department'], method='counts_percentage', output='return')
    assert 'Counts' in result and 'Percentages' in result
    assert 'HR' in result and 'Tech' in result and 'Admin' in result


def test_unique_values_method(complex_categorical_df):
    """
    Verify that unique values are correctly listed in the output.
    """
    result = explore_cat(complex_categorical_df, ['Department'], method='unique_values', output='return')
    assert 'HR' in result and 'Tech' in result and 'Admin' in result


def test_all_methods(complex_categorical_df):
    """
    Ensure the 'all' method integrates outputs of all individual methods correctly.
    """
    result = explore_cat(complex_categorical_df, ['Department'], method='all', output='return')
    assert 'UNIQUE VALUES' in result
    assert 'COUNTS & PERCENTAGE' in result
    assert 'ENTROPY' in result
