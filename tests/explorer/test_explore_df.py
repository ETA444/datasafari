import pytest
import pandas as pd
from datasafari.explorer.explore_df import explore_df

@pytest.fixture
def sample_df():
    """Create a sample DataFrame to use in the tests."""
    data = {
        'A': [1, 2, 3, None, 5],
        'B': [5, 6, None, 8, 9],
        'C': ['foo', 'bar', 'baz', 'qux', 'quux']
    }
    return pd.DataFrame(data)


def test_explore_df_invalid_df_type():
    """Test passing invalid df type should raise TypeError."""
    with pytest.raises(TypeError):
        explore_df("not a dataframe", 'all')


def test_explore_df_invalid_method(sample_df):
    """Test passing invalid method should raise ValueError."""
    with pytest.raises(ValueError):
        explore_df(sample_df, 'invalid_method')


def test_explore_df_invalid_output(sample_df):
    """Test passing invalid output should raise ValueError."""
    with pytest.raises(ValueError):
        explore_df(sample_df, 'all', output='invalid_output')


def test_explore_df_method_all_print(sample_df):
    """Test method 'all's outputs being correct."""
    result = explore_df(sample_df, 'all', output='return')
    assert 'DESCRIBE' in result
    assert 'HEAD' in result
    assert 'INFO' in result
    assert 'NA_COUNT' in result


def test_explore_df_method_describe_print(sample_df):
    """Test method 'described' and that no other methods are in the output."""
    result = explore_df(sample_df, 'desc', output='return')
    assert 'DESCRIBE' in result
    assert 'HEAD' not in result
    assert 'INFO' not in result
    assert 'NA_COUNT' not in result


def test_explore_df_custom_method_desc(sample_df):
    """Test method 'desc' with custom percentiles."""
    result = explore_df(sample_df, 'desc', output='return', percentiles=[0.05, 0.95])
    assert '5%' in result
    assert '95%' in result


def test_explore_df_head_with_zero_rows(sample_df):
    """Test head method with n=0 should return empty head."""
    result = explore_df(sample_df, 'head', output='return', n=0)
    assert result == "<<______HEAD______>>\nEmpty DataFrame\nColumns: [A, B, C]\nIndex: []\n"


def test_explore_df_empty_dataframe():
    """Test handling of an empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        explore_df(empty_df, 'all')


def test_explore_df_info_verbose(sample_df):
    """Test info method with verbose output."""
    result = explore_df(sample_df, 'info', verbose=True, output='return')
    assert 'RangeIndex:' in result
    assert 'Data columns (total 3 columns):' in result


def test_explore_df_na(sample_df):
    """Test NA method output correctness."""
    result = explore_df(sample_df, 'na', output='return')
    assert "A    1\nB    1\nC    0" in result
    assert "A    20.0\nB    20.0\nC    0.0" in result


def test_explore_df_all_methods_with_kwargs(sample_df):
    result = explore_df(sample_df, 'all', n=3, percentiles=[0.1, 0.9], verbose=False, output='return')
    assert '5%' not in result  # Check that no default percentiles are applied
    assert '90%' in result  # Check that passed percentiles are applied
    assert 'head' in result
    assert 'info' in result
    assert 'na' in result