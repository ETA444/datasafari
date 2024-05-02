import pytest
import pandas as pd
import numpy as np
from datasafari.evaluator.evaluate_variance import evaluate_variance


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Group': np.random.choice(['A', 'B', 'C'], 100),
        'Data': np.random.normal(0, 1, 100)
    })


# TESTING ERROR-HANDLING #
def test_evaluate_variance_non_dataframe_input():
    """ Test that non-DataFrame input raises a TypeError. """
    with pytest.raises(TypeError, match="The 'df' parameter must be a pandas DataFrame."):
        evaluate_variance("not a dataframe", 'Data', 'Group')


def test_evaluate_variance_non_string_target_variable(sample_data):
    """ Test that non-string target_variable raises a TypeError. """
    with pytest.raises(TypeError, match="The 'target_variable' parameter must be a string."):
        evaluate_variance(sample_data, 123, 'Group')


def test_evaluate_variance_non_string_grouping_variable(sample_data):
    """ Test that non-string grouping_variable raises a TypeError. """
    with pytest.raises(TypeError, match="The 'grouping_variable' parameter must be a string."):
        evaluate_variance(sample_data, 'Data', 123)


def test_evaluate_variance_non_boolean_normality_info(sample_data):
    """ Test that non-boolean normality_info raises a TypeError. """
    with pytest.raises(TypeError, match="The 'normality_info' parameter must be a boolean if provided."):
        evaluate_variance(sample_data, 'Data', 'Group', normality_info='yes')


def test_evaluate_variance_non_string_method(sample_data):
    """ Test that non-string method raises a TypeError. """
    with pytest.raises(TypeError, match="The 'method' parameter must be a string."):
        evaluate_variance(sample_data, 'Data', 'Group', method=123)


def test_evaluate_variance_non_boolean_pipeline(sample_data):
    """ Test that non-boolean pipeline raises a TypeError. """
    with pytest.raises(TypeError, match="The 'pipeline' parameter must be a boolean."):
        evaluate_variance(sample_data, 'Data', 'Group', pipeline='yes')


def test_evaluate_variance_empty_dataframe():
    """ Test handling of empty DataFrame. """
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError, match="The input DataFrame is empty."):
        evaluate_variance(df_empty, 'Data', 'Group')


def test_evaluate_variance_non_existent_target_variable(sample_data):
    """ Test handling of non-existent target_variable. """
    with pytest.raises(ValueError, match="The target variable 'NonExistentVar' was not found in the DataFrame."):
        evaluate_variance(sample_data, 'NonExistentVar', 'Group')


def test_evaluate_variance_non_existent_grouping_variable(sample_data):
    """ Test handling of non-existent grouping_variable. """
    with pytest.raises(ValueError, match="The grouping variable 'NonExistentVar' was not found in the DataFrame."):
        evaluate_variance(sample_data, 'Data', 'NonExistentVar')


def test_evaluate_variance_invalid_method(sample_data):
    """ Test handling of invalid method input. """
    with pytest.raises(ValueError, match="The method 'invalid_method' is not supported."):
        evaluate_variance(sample_data, 'Data', 'Group', method='invalid_method')


def test_evaluate_variance_invalid_target_variable_type(sample_data):
    """ Test if target_variable is not numerical. """
    sample_data['Data'] = sample_data['Data'].astype(str)
    with pytest.raises(ValueError, match="The target variable 'Data' must be a numerical variable."):
        evaluate_variance(sample_data, 'Data', 'Group')


def test_evaluate_variance_invalid_grouping_variable_type(sample_data):
    """ Test if grouping_variable is not categorical. """
    sample_data['Group'] = np.linspace(0, 1, 100)
    with pytest.raises(ValueError, match="The grouping variable 'Group' must be a categorical variable."):
        evaluate_variance(sample_data, 'Data', 'Group')


# TESTING FUNCTIONALITY #

def test_evaluate_variance_levene(sample_data):
    """ Test the Levene test functionality in evaluate_variance. """
    result = evaluate_variance(sample_data, 'Data', 'Group', method='levene')
    assert 'levene' in result
    assert isinstance(result['levene'], dict)
    assert 'stat' in result['levene'] and 'p' in result['levene']


def test_evaluate_variance_bartlett(sample_data):
    """ Test the Bartlett test functionality in evaluate_variance with normality assumed. """
    result = evaluate_variance(sample_data, 'Data', 'Group', normality_info=True, method='bartlett')
    assert 'bartlett' in result
    assert isinstance(result['bartlett'], dict)
    assert 'stat' in result['bartlett'] and 'p' in result['bartlett']


def test_evaluate_variance_fligner(sample_data):
    """ Test the Fligner-Killeen test functionality in evaluate_variance. """
    result = evaluate_variance(sample_data, 'Data', 'Group', method='fligner')
    assert 'fligner' in result
    assert isinstance(result['fligner'], dict)
    assert 'stat' in result['fligner'] and 'p' in result['fligner']


def test_evaluate_variance_consensus_all_agree(sample_data):
    """ Test the consensus method when all tests agree on the result. """
    sample_data['Data'] = np.random.normal(0, 1, 100)  # Ensure data is normal for consistency
    result = evaluate_variance(sample_data, 'Data', 'Group', normality_info=True, method='consensus')
    assert isinstance(result, dict)
    assert all(test in result for test in ['levene', 'bartlett', 'fligner'])
    consensus = [result[test]['equal_variances'] for test in result]
    assert all(consensus) or not any(consensus)  # Check if all are True or all are False


def test_evaluate_variance_pipeline_mode(sample_data):
    """ Test the pipeline mode returns a simple boolean for consensus. """
    result = evaluate_variance(sample_data, 'Data', 'Group', pipeline=True)
    assert isinstance(result, bool)
