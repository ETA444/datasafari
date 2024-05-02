import pytest
import pandas as pd
import numpy as np
from datasafari.explorer.explore_num import explore_num


@pytest.fixture
def sample_numerical_df():
    """Provide a DataFrame with numerical data for testing."""
    np.random.seed(0)
    return pd.DataFrame({
        'Feature1': np.random.normal(0, 1, 100),
        'Feature2': np.random.exponential(1, 100),
        'Feature3': np.random.randint(1, 100, 100)
    })


# TESTING ERROR-HANDLING #
def test_non_dataframe_input():
    """
    Test if the input df is not a DataFrame to ensure TypeError is raised.
    """
    with pytest.raises(TypeError, match="The df parameter must be a pandas DataFrame."):
        explore_num("not a dataframe", ['Feature1'], method='all')


def test_non_list_numerical_variables(sample_numerical_df):
    """
    Test if the numerical_variables is not a list to ensure TypeError is raised.
    """
    with pytest.raises(TypeError, match="must be a list of variable names"):
        explore_num(sample_numerical_df, 'Feature1')


def test_non_string_in_numerical_variables(sample_numerical_df):
    """
    Test if elements of numerical_variables are not strings to ensure TypeError is raised.
    """
    with pytest.raises(TypeError, match="All items in the numerical_variables list must be strings representing column names."):
        explore_num(sample_numerical_df, [123])


def test_non_string_method(sample_numerical_df):
    """
    Test if method is not a string to ensure TypeError is raised.
    """
    with pytest.raises(TypeError, match="The method parameter must be a string."):
        explore_num(sample_numerical_df, ['Feature1'], method=123)


def test_non_string_output(sample_numerical_df):
    """
    Test if output is not a string to ensure TypeError is raised.
    """
    with pytest.raises(TypeError, match="The output parameter must be a string."):
        explore_num(sample_numerical_df, ['Feature1'], output=123)


def test_non_number_threshold_z(sample_numerical_df):
    """
    Test if threshold_z is not a float or int to ensure TypeError is raised.
    """
    with pytest.raises(TypeError, match="The value of threshold_z must be a float or int."):
        explore_num(sample_numerical_df, ['Feature1'], method='outliers_zscore', threshold_z="high")


def test_empty_dataframe():
    """
    Test handling of an empty DataFrame to ensure ValueError is raised.
    """
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="The input DataFrame is empty."):
        explore_num(df, ['Feature1'])


def test_invalid_method(sample_numerical_df):
    """
    Test if method is not valid to ensure ValueError is raised.
    """
    with pytest.raises(ValueError, match="Invalid method"):
        explore_num(sample_numerical_df, ['Feature1'], method='invalid_method')


def test_invalid_output_method(sample_numerical_df):
    """
    Test if output method is not valid to ensure ValueError is raised.
    """
    with pytest.raises(ValueError, match="Invalid output method. Choose 'print' or 'return'."):
        explore_num(sample_numerical_df, ['Feature1'], output='invalid_output')


def test_empty_numerical_variables_list(sample_numerical_df):
    """
    Test if numerical_variables list is empty to ensure ValueError is raised.
    """
    with pytest.raises(ValueError, match="The 'numerical_variables' list must contain at least one column name."):
        explore_num(sample_numerical_df, [])


def test_non_numerical_variables(sample_numerical_df):
    """
    Test if variables provided are not numerical to ensure ValueError is raised.
    """
    sample_numerical_df['NonNumeric'] = ['a', 'b', 'c'] * 33 + ['c']
    with pytest.raises(ValueError, match="The 'numerical_variables' list must contain only"):
        explore_num(sample_numerical_df, ['NonNumeric'])


def test_missing_numerical_variables(sample_numerical_df):
    """
    Test if specified variables are not in DataFrame to ensure ValueError is raised.
    """
    with pytest.raises(ValueError, match="MissingFeature"):
        explore_num(sample_numerical_df, ['MissingFeature'])


# TESTING FUNCTIONALITY #
def test_correlation_analysis(sample_numerical_df):
    """Verify correlation analysis computes and returns correlation matrices correctly."""
    result = explore_num(sample_numerical_df, ['Feature1', 'Feature2', 'Feature3'], method='correlation_analysis', output='return')
    assert isinstance(result, list)
    assert all(isinstance(df, pd.DataFrame) for df in result)
    assert 'Feature1' in result[0].columns
    assert 'Feature2' in result[0].columns


def test_distribution_analysis(sample_numerical_df):
    """Test distribution analysis to ensure it computes descriptive statistics and normality tests."""
    result = explore_num(sample_numerical_df, ['Feature1', 'Feature2', 'Feature3'], method='distribution_analysis', output='return')
    assert isinstance(result, pd.DataFrame)
    assert 'mean' in result.columns
    assert 'std_dev' in result.columns


def test_outliers_zscore(sample_numerical_df):
    """Check Z-score outlier detection correctly identifies outliers and returns results."""
    result_dict, result_df = explore_num(sample_numerical_df, ['Feature1', 'Feature2'], method='outliers_zscore', output='return', threshold_z=2)
    assert isinstance(result_dict, dict)
    assert isinstance(result_df, pd.DataFrame)
    assert 'Feature1' in result_dict
    assert not result_df.empty


def test_outliers_iqr(sample_numerical_df):
    """Ensure IQR method detects outliers and formats the output correctly."""
    result_dict, result_df = explore_num(sample_numerical_df, ['Feature1', 'Feature2'], method='outliers_iqr', output='return')
    assert isinstance(result_dict, dict)
    assert isinstance(result_df, pd.DataFrame)
    assert 'Feature1' in result_dict
    assert not result_df.empty


def test_outliers_mahalanobis(sample_numerical_df):
    """Verify Mahalanobis distance correctly identifies outliers."""
    result_df = explore_num(sample_numerical_df, ['Feature1', 'Feature2', 'Feature3'], method='outliers_mahalanobis', output='return')
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty


def test_multicollinearity(sample_numerical_df):
    """Test multicollinearity detection outputs VIF scores correctly."""
    result = explore_num(sample_numerical_df, ['Feature1', 'Feature2', 'Feature3'], method='multicollinearity', output='return')
    assert isinstance(result, str)
    assert 'Feature1    1.010043' in result
    assert 'Feature2    1.013722' in result
    assert 'Feature3    1.006570' in result


def test_all_methods(sample_numerical_df):
    """Check that 'all' method aggregates outputs from all individual methods."""
    result = explore_num(sample_numerical_df, ['Feature1', 'Feature2', 'Feature3'], method='all', output='return')
    assert isinstance(result, str)
    assert 'CORRELATIONS' in result
    assert 'DISTRIBUTION ANALYSIS' in result
    assert 'OUTLIERS' in result
    assert 'MULTICOLLINEARITY CHECK' in result
