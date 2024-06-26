import pytest
import pandas as pd
from datasafari.explorer.explore_df import explore_df


@pytest.fixture
def sample_df():
    """
    Provide a DataFrame with mixed types for testing.
    """
    data = {
        'A': [1, 2, 3, None, 5],
        'B': [5, 6, None, 8, 9],
        'C': ['foo', 'bar', 'baz', 'qux', 'quux']
    }
    return pd.DataFrame(data)


# TEST ERROR-HANDLING #
def test_explore_df_invalid_df_type():
    """
    Ensure passing an invalid df type raises TypeError.
    """
    with pytest.raises(TypeError, match="The df parameter must be a pandas DataFrame."):
        explore_df("not a dataframe", 'all')


def test_explore_df_invalid_method_type(sample_df):
    """
    Ensure passing a non-string method raises TypeError.
    """
    with pytest.raises(TypeError, match="The method parameter must be a string."):
        explore_df(sample_df, method=123)


def test_explore_df_invalid_output_type(sample_df):
    """
    Ensure passing a non-string output raises TypeError.
    """
    with pytest.raises(TypeError, match="The output parameter must be a string."):
        explore_df(sample_df, output=123)


def test_explore_df_invalid_method_name(sample_df):
    """
    Ensure passing an invalid method name raises ValueError.
    """
    with pytest.raises(ValueError, match="Invalid method 'invalid_method'. Valid options are: na, desc, head, info, all."):
        explore_df(sample_df, 'invalid_method')


def test_explore_df_invalid_output_value(sample_df):
    """
    Ensure passing an invalid output value raises ValueError.
    """
    with pytest.raises(ValueError, match="Invalid output method. Choose 'print' or 'return'."):
        explore_df(sample_df, 'all', output='something')


def test_explore_df_empty_dataframe():
    """
    Test handling of an empty DataFrame.
    """
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="The input DataFrame is empty."):
        explore_df(empty_df, 'all')


def test_explore_df_unsupported_info_kwarg(sample_df):
    """
    Test passing unsupported kwargs to the 'info' method.
    """
    with pytest.raises(ValueError, match="'buf' parameter is not supported in the 'info' method within explore_df."):
        explore_df(sample_df, 'info', buf=None)


# TESTING FUNCTIONALITY #
def test_explore_df_method_all_print(sample_df):
    """
    Check that method 'all' correctly integrates outputs from all methods.
    """
    result = explore_df(sample_df, 'all', output='return')
    assert 'DESCRIBE' in result
    assert 'HEAD' in result
    assert 'INFO' in result
    assert 'NA_COUNT' in result


def test_explore_df_custom_method_desc(sample_df):
    """
    Test 'desc' method with custom percentiles.
    """
    result = explore_df(sample_df, 'desc', output='return', percentiles=[0.05, 0.95])
    assert '5%' in result
    assert '95%' in result


def test_explore_df_head_with_zero_rows(sample_df):
    """
    Ensure head method with n=0 returns an empty DataFrame description.
    """
    result = explore_df(sample_df, 'head', output='return', n=0)
    assert result == "<______HEAD______>\nEmpty DataFrame\nColumns: [A, B, C]\nIndex: []\n"


def test_explore_df_info_verbose(sample_df):
    """
    Test 'info' method with verbose=True provides detailed output.
    """
    result = explore_df(sample_df, 'info', verbose=True, output='return')
    assert 'RangeIndex:' in result
    assert 'Data columns (total 3 columns):' in result


def test_explore_df_na(sample_df):
    """
    Test 'na' method correctly calculates and formats NA counts and percentages.
    """
    result = explore_df(sample_df, 'na', output='return')
    assert "1" in result
    assert "0" in result
    assert "20.0" in result
    assert "0.0" in result


def test_explore_df_all_methods_with_kwargs(sample_df):
    """
    Test 'all' method respects kwargs and outputs correctly.
    """
    result = explore_df(sample_df, 'all', n=3, percentiles=[0.1, 0.9], verbose=False, output='return')
    assert '5%' not in result  # Check that no default percentiles are applied
    assert '90%' in result  # Check that passed percentiles are applied
    assert 'HEAD' in result
    assert 'INFO' in result
    assert 'NA' in result
