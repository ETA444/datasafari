import pytest
import pandas as pd
import numpy as np
from datasafari.evaluator.evaluate_normality import evaluate_normality


@pytest.fixture
def sample_normality_df():
    """Creates a sample DataFrame to use in normality tests."""
    np.random.seed(0)
    return pd.DataFrame({
        'Group': np.random.choice(['A', 'B', 'C'], size=100),
        'NumericData': np.random.normal(loc=20, scale=5, size=100),
        'CategoricalData': np.random.choice(['X', 'Y'], size=100),
        'NonNumericData': ['text'] * 100
    })


# TESTING ERROR-HANDLING #

def test_invalid_dataframe_type():
    """Test passing non-DataFrame as df should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_normality("not a dataframe", 'NumericData', 'Group')


def test_invalid_target_variable_type(sample_normality_df):
    """Test passing non-string as target_variable should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_normality(sample_normality_df, 123, 'Group')


def test_invalid_grouping_variable_type(sample_normality_df):
    """Test passing non-string as grouping_variable should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_normality(sample_normality_df, 'NumericData', 123)


def test_invalid_method_type(sample_normality_df):
    """Test passing non-string as method should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_normality(sample_normality_df, 'NumericData', 'Group', method=123)


def test_invalid_pipeline_type(sample_normality_df):
    """Test passing non-boolean as pipeline should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_normality(sample_normality_df, 'NumericData', 'Group', pipeline='yes')


def test_empty_dataframe():
    """Test handling of an empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        evaluate_normality(empty_df, 'NumericData', 'Group')


def test_missing_target_variable(sample_normality_df):
    """Test handling of a missing target variable."""
    with pytest.raises(ValueError):
        evaluate_normality(sample_normality_df, 'MissingData', 'Group')


def test_missing_grouping_variable(sample_normality_df):
    """Test handling of a missing grouping variable."""
    with pytest.raises(ValueError):
        evaluate_normality(sample_normality_df, 'NumericData', 'MissingGroup')


def test_non_numerical_target_variable(sample_normality_df):
    """Test handling of a non-numerical target variable."""
    with pytest.raises(ValueError):
        evaluate_normality(sample_normality_df, 'NonNumericData', 'Group')


def test_non_categorical_grouping_variable(sample_normality_df):
    """Test handling of a non-categorical grouping variable."""
    with pytest.raises(ValueError):
        evaluate_normality(sample_normality_df, 'NumericData', 'NumericData')


def test_invalid_method_specified(sample_normality_df):
    """Test handling of an invalid method specification."""
    with pytest.raises(ValueError):
        evaluate_normality(sample_normality_df, 'NumericData', 'Group', method='unknown_method')
