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


def test_empty_dataframe():
    """ Test handling of empty DataFrame """
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="The input DataFrame is empty."):
        explore_cat(df, ['Category'])


def test_non_existent_column(complex_categorical_df):
    """ Test handling of non-existent column names """
    with pytest.raises(ValueError):
        explore_cat(complex_categorical_df, ['NonExistentColumn'])


def test_non_categorical_column(complex_categorical_df):
    """ Test handling of non-categorical columns """
    with pytest.raises(ValueError, match="The 'categorical_variables' list must contain only names"):
        explore_cat(complex_categorical_df, ['Age', 'Department'])


def test_entropy_method(complex_categorical_df):
    """ Test entropy calculation """
    result = explore_cat(complex_categorical_df, ['Department'], method='entropy', output='return')
    assert 'Entropy' in result
    assert '1.585' in result


def test_invalid_method(complex_categorical_df):
    """ Test invalid method input """
    with pytest.raises(ValueError):
        explore_cat(complex_categorical_df, ['Department'], method='invalid_method')


def test_counts_percentage_output(complex_categorical_df):
    """ Test counts and percentage output correctness """
    result = explore_cat(complex_categorical_df, ['Department'], method='counts_percentage', output='return')
    assert 'Counts' in result and 'Percentages' in result
    assert 'HR' in result and 'Tech' in result and 'Admin' in result


def test_unique_values_method(complex_categorical_df):
    """ Test unique values method """
    result = explore_cat(complex_categorical_df, ['Department'], method='unique_values', output='return')
    assert 'HR' in result and 'Tech' in result and 'Admin' in result


def test_all_methods(complex_categorical_df):
    """ Test 'all' method integrates outputs correctly """
    result = explore_cat(complex_categorical_df, ['Department'], method='all', output='return')
    assert 'UNIQUE VALUES' in result and 'COUNTS & PERCENTAGE' in result and 'ENTROPY' in result
