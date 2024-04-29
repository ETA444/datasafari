import pytest
import pandas as pd
from datasafari.explorer.explore_df import explore_df


@pytest.fixture
def sample_df():
    """Create a sample df to use in all test_explore_df_...() tests"""
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