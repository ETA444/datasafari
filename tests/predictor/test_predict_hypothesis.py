import pytest
import numpy as np
import pandas as pd
from datasafari.predictor.predict_hypothesis import hypothesis_predictor_core_n, hypothesis_predictor_core_c, predict_hypothesis


@pytest.fixture
def sample_data():
    """ Provides sample data for tests. """
    return pd.DataFrame({
        'Group': np.random.choice(['A', 'B', 'C'], 100),
        'Value': np.random.normal(0, 1, 100)
    })


# TESTING ERROR-HANDLING for hypothesis_predictor_core_n() #

def test_hypothesis_predictor_core_n_non_dataframe_input():
    """ Test that non-DataFrame input raises a TypeError. """
    with pytest.raises(TypeError, match="pandas DataFrame"):
        hypothesis_predictor_core_n("not_a_dataframe", 'Value', 'Group', True, True)


def test_hypothesis_predictor_core_n_non_string_target_variable(sample_data):
    """ Test that non-string target_variable raises a TypeError. """
    with pytest.raises(TypeError, match="must be a string"):
        hypothesis_predictor_core_n(sample_data, 123, 'Group', True, True)


def test_hypothesis_predictor_core_n_non_string_grouping_variable(sample_data):
    """ Test that non-string grouping_variable raises a TypeError. """
    with pytest.raises(TypeError, match="must be a string"):
        hypothesis_predictor_core_n(sample_data, 'Value', 123, True, True)


def test_hypothesis_predictor_core_n_non_boolean_normality_bool(sample_data):
    """ Test that non-boolean normality_bool raises a TypeError. """
    with pytest.raises(TypeError, match="must be a boolean"):
        hypothesis_predictor_core_n(sample_data, 'Value', 'Group', 'yes', True)


def test_hypothesis_predictor_core_n_non_boolean_equal_variances_bool(sample_data):
    """ Test that non-boolean equal_variances_bool raises a TypeError. """
    with pytest.raises(TypeError, match="must be a boolean"):
        hypothesis_predictor_core_n(sample_data, 'Value', 'Group', True, 'yes')


def test_hypothesis_predictor_core_n_empty_dataframe():
    """ Test that empty DataFrame raises a ValueError. """
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError, match="input DataFrame is empty"):
        hypothesis_predictor_core_n(df_empty, 'Value', 'Group', True, True)


def test_hypothesis_predictor_core_n_non_existent_target_variable(sample_data):
    """ Test that non-existent target_variable raises a ValueError. """
    with pytest.raises(ValueError, match="target variable 'NonExistentVar'"):
        hypothesis_predictor_core_n(sample_data, 'NonExistentVar', 'Group', True, True)


def test_hypothesis_predictor_core_n_non_existent_grouping_variable(sample_data):
    """ Test that non-existent grouping_variable raises a ValueError. """
    with pytest.raises(ValueError, match="grouping variable 'NonExistentVar'"):
        hypothesis_predictor_core_n(sample_data, 'Value', 'NonExistentVar', True, True)


def test_hypothesis_predictor_core_n_non_numerical_target_variable(sample_data):
    """ Test that non-numerical target_variable raises a ValueError. """
    sample_data['NonNumeric'] = ['A'] * 100
    with pytest.raises(ValueError, match="must be a numerical variable"):
        hypothesis_predictor_core_n(sample_data, 'NonNumeric', 'Group', True, True)


def test_hypothesis_predictor_core_n_non_categorical_grouping_variable(sample_data):
    """ Test that non-categorical grouping_variable raises a ValueError. """
    sample_data['NonCategorical'] = np.linspace(0, 1, 100)
    with pytest.raises(ValueError, match="must be a categorical variable"):
        hypothesis_predictor_core_n(sample_data, 'Value', 'NonCategorical', True, True)
