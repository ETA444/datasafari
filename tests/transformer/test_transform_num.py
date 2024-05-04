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


# TESTING FUNCTIONALITY #

def test_transform_num_standardize(sample_data):
    """ Test standardizing numerical data. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature1', 'Feature2'], 'standardize')
    assert transformed_columns.mean().round(1).all() == 0
    assert transformed_columns.std().round(1).all() == 1


def test_transform_num_log(sample_data):
    """ Test logarithm transformation. """
    sample_data['Feature2'] += 1  # Ensure all values are positive for log transformation
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature2'], 'log')
    assert (transformed_columns >= 0).all().all()


def test_transform_num_normalize(sample_data):
    """ Test normalizing numerical data. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature1'], 'normalize')
    assert transformed_columns.min().all() == 0
    assert transformed_columns.max().all() == 1


def test_transform_num_quantile_normal(sample_data):
    """ Test quantile transformation to normal distribution. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature2'], 'quantile', output_distribution='normal')
    assert (transformed_columns.mean().round(1).all() == 0) and (transformed_columns.std().round(1).all() == 1)


def test_transform_num_quantile_uniform(sample_data):
    """ Test quantile transformation to uniform distribution. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature2'], 'quantile', output_distribution='uniform')
    assert transformed_columns.min().all() == 0
    assert transformed_columns.max().all() == 1


def test_transform_num_robust(sample_data):
    """ Test robust scaling. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature1', 'Feature3'], 'robust', quantile_range=(25.0, 75.0))
    assert transformed_columns.median().all() == 0


def test_transform_num_boxcox(sample_data):
    """ Test Box-Cox transformation. """
    sample_data['Feature2'] += 1  # Ensure all values are positive for Box-Cox
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature2'], 'boxcox')
    assert transformed_columns.skew().abs().all() < 2


def test_transform_num_yeojohnson(sample_data):
    """ Test Yeo-Johnson transformation. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature1'], 'yeojohnson')
    assert transformed_columns.skew().abs().all() < 2


def test_transform_num_power(sample_data):
    """ Test power transformation with global power value. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature1'], 'power', power=2)
    assert (transformed_columns >= 0).all().all()


def test_transform_num_winsorization(sample_data):
    """ Test winsorization transformation. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature1'], 'winsorization', lower_percentile=0.05, upper_percentile=0.95)
    assert transformed_columns.max().all() <= transformed_df['Feature1'].max()
    assert transformed_columns.min().all() >= transformed_df['Feature1'].min()


def test_transform_num_interaction(sample_data):
    """ Test interaction terms creation. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature1', 'Feature2'], 'interaction', interaction_pairs=[('Feature1', 'Feature2')])
    assert 'interaction_Feature1_x_Feature2' in transformed_columns.columns


def test_transform_num_polynomial(sample_data):
    """ Test polynomial features generation with global degree. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature1'], 'polynomial', degree=2)
    assert 'Feature1_degree_2' in transformed_columns.columns


def test_transform_num_bin(sample_data):
    """ Test binning transformation with global bin count. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature3'], 'bin', bins=5)
    assert transformed_columns['Feature3'].nunique() == 5


def test_transform_num_bin_custom_bins(sample_data):
    """ Test binning transformation with custom bin edges. """
    transformed_df, transformed_columns = transform_num(sample_data, ['Feature3'], 'bin', bin_map={'Feature3': {'edges': [0, 25, 50, 75, 100]}})
    assert transformed_columns['Feature3'].nunique() == 4
