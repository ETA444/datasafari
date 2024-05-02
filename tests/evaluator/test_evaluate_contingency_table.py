import pytest
import pandas as pd
import numpy as np
from datasafari.evaluator.evaluate_contingency_table import evaluate_contingency_table


@pytest.fixture
def valid_data():
    """Valid contingency table for testing."""
    data = np.array([[10, 15], [20, 25]])
    return pd.DataFrame(data)


# TESTING ERROR-HANDLING #

def test_invalid_contingency_table_type():
    """Test passing invalid contingency_table type should raise TypeError."""
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        evaluate_contingency_table("not a dataframe")


def test_invalid_min_sample_size_yates_type(valid_data):
    """Test passing invalid min_sample_size_yates type should raise TypeError."""
    with pytest.raises(TypeError, match="must be an integer"):
        evaluate_contingency_table(valid_data, min_sample_size_yates="forty")


def test_invalid_pipeline_type(valid_data):
    """Test passing invalid pipeline type should raise TypeError."""
    with pytest.raises(TypeError, match="must be a boolean"):
        evaluate_contingency_table(valid_data, pipeline="yes")


def test_invalid_quiet_type(valid_data):
    """Test passing invalid quiet type should raise TypeError."""
    with pytest.raises(TypeError, match="must be a boolean"):
        evaluate_contingency_table(valid_data, quiet="no")


def test_empty_contingency_table():
    """Test passing an empty contingency table should raise ValueError."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_contingency_table(df)


def test_negative_min_sample_size_yates(valid_data):
    """Test passing a negative min_sample_size_yates should raise ValueError."""
    with pytest.raises(ValueError, match="must be a positive integer"):
        evaluate_contingency_table(valid_data, min_sample_size_yates=-1)


def test_zero_min_sample_size_yates(valid_data):
    """Test passing zero for min_sample_size_yates should raise ValueError."""
    with pytest.raises(ValueError, match="must be a positive integer"):
        evaluate_contingency_table(valid_data, min_sample_size_yates=0)


# TESTING FUNCTIONALITY #

def test_functionality_large_sample_without_yates_correction(valid_data):
    """Verify that Yates' correction is not recommended for large sample sizes."""
    # Modify the fixture data to simulate a large sample size scenario
    large_sample_data = valid_data * 100  # Increasing counts to simulate larger sample
    results = evaluate_contingency_table(large_sample_data, min_sample_size_yates=40, pipeline=True)
    assert not results[1]  # Yates' correction should not be applied


def test_functionality_small_sample_with_yates_correction():
    """Verify that Yates' correction is recommended for small 2x2 tables."""
    small_data = pd.DataFrame([[5, 5], [10, 15]])  # Small counts in a 2x2 table
    results = evaluate_contingency_table(small_data, min_sample_size_yates=40, pipeline=True)
    assert results[1]  # Yates' correction should be applied


def test_functionality_non_2x2_table():
    """Ensure exact tests are not recommended for tables not 2x2."""
    non_2x2_data = pd.DataFrame(np.random.randint(0, 100, size=(3, 3)))
    results = evaluate_contingency_table(non_2x2_data, pipeline=True)
    _, _, barnard, boschloo, fisher = results
    assert not barnard and not boschloo and not fisher  # No exact tests are viable


def test_functionality_2x2_table():
    """Ensure all exact tests are recommended for 2x2 tables."""
    data_2x2 = pd.DataFrame([[10, 15], [20, 25]])
    results = evaluate_contingency_table(data_2x2, pipeline=True)
    _, _, barnard, boschloo, fisher = results
    assert barnard and boschloo and fisher  # All exact tests should be viable


def test_functionality_output_dictionary_format(valid_data):
    """Test if the output format as a dictionary is correct when pipeline is False."""
    results = evaluate_contingency_table(valid_data, pipeline=False)
    expected_keys = ['chi2_contingency', 'yates_correction', 'barnard_exact', 'boschloo_exact', 'fisher_exact']
    assert isinstance(results, dict)
    assert all(key in results for key in expected_keys)
    assert all(type(value) == bool for value in results.values())


def test_functionality_output_tuple_format(valid_data):
    """Test if the output format as a tuple is correct when pipeline is True."""
    results = evaluate_contingency_table(valid_data, pipeline=True)
    assert isinstance(results, tuple)
    assert len(results) == 5
    assert all(type(value) == bool for value in results)
