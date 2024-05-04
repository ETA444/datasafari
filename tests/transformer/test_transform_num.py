import pytest
import pandas as pd
import numpy as np
from datasafari.transformer.transform_num import transform_num


@pytest.fixture
def sample_data():
    """ Provides sample data for tests. """
    return pd.DataFrame({
        'Feature1': np.random.normal(0, 1, 100),
        'Feature2': np.random.exponential(1, 100),
        'Feature3': np.random.randint(1, 100, 100)
    })


# TESTING ERROR-HANDLING #

def test_transform_num_invalid_df_type():
    """ Test that non-DataFrame input raises a TypeError. """
    with pytest.raises(TypeError):
        transform_num("not_a_dataframe", ['Feature1'], 'standardize')


def test_transform_num_invalid_numerical_variables_type():
    """ Test that non-list numerical_variables raises a TypeError. """
    with pytest.raises(TypeError):
        transform_num(pd.DataFrame(), 'Feature1', 'standardize')


def test_transform_num_invalid_numerical_variable_entries(sample_data):
    """ Test that non-string entries in numerical_variables list raises a TypeError. """
    with pytest.raises(TypeError):
        transform_num(sample_data, [1, 2, 3], 'standardize')


def test_transform_num_invalid_method_type(sample_data):
    """ Test that non-string method raises a TypeError. """
    with pytest.raises(TypeError):
        transform_num(sample_data, ['Feature1'], 123)


def test_transform_num_empty_dataframe():
    """ Test handling of empty DataFrame. """
    with pytest.raises(ValueError):
        transform_num(pd.DataFrame(), ['Feature1'], 'standardize')


def test_transform_num_nonexistent_numerical_variable(sample_data):
    """ Test handling of nonexistent numerical variable. """
    with pytest.raises(ValueError):
        transform_num(sample_data, ['Nonexistent'], 'standardize')


def test_transform_num_empty_numerical_variables_list(sample_data):
    """ Test handling of empty numerical_variables list. """
    with pytest.raises(ValueError):
        transform_num(sample_data, [], 'standardize')


def test_transform_num_invalid_quantile_parameters(sample_data):
    """ Test invalid quantile parameters like n_quantiles and quantile_range. """
    with pytest.raises(TypeError):
        transform_num(sample_data, ['Feature1'], 'quantile', n_quantiles='1000')  # Invalid type for n_quantiles
    with pytest.raises(ValueError):
        transform_num(sample_data, ['Feature1'], 'quantile', n_quantiles=-1000)  # Negative n_quantiles


def test_transform_num_invalid_quantile_range(sample_data):
    """ Test invalid quantile range input. """
    with pytest.raises(TypeError):
        transform_num(sample_data, ['Feature1'], 'robust', quantile_range=(0.25, '75%'))  # Non-float range values
    with pytest.raises(ValueError):
        transform_num(sample_data, ['Feature1'], 'robust', quantile_range=(0.75, 0.25))  # Lower range greater than upper range


def test_transform_num_invalid_interaction_pairs(sample_data):
    """ Test invalid interaction pairs format. """
    with pytest.raises(TypeError):
        transform_num(sample_data, ['Feature1', 'Feature2'], 'interaction', interaction_pairs='not_a_list')


def test_transform_num_invalid_power_parameter(sample_data):
    """ Test invalid power parameter. """
    with pytest.raises(TypeError):
        transform_num(sample_data, ['Feature1'], 'power', power='not_a_float')


def test_transform_num_invalid_power_map(sample_data):
    """ Test invalid power map dictionary. """
    with pytest.raises(TypeError):
        transform_num(sample_data, ['Feature1'], 'power', power_map='not_a_dict')


def test_transform_num_invalid_bin_parameters(sample_data):
    """ Test invalid bins parameter. """
    with pytest.raises(TypeError):
        transform_num(sample_data, ['Feature1'], 'bin', bins='not_an_int')
