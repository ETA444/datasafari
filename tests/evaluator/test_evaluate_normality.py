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


# TESTING FUNCTIONALITY #

def test_normality_consensus_method(sample_normality_df):
    """Test the consensus method to evaluate normality across all methods."""
    results = evaluate_normality(sample_normality_df, 'NumericData', 'Group', method='consensus', pipeline=False)
    assert isinstance(results, dict)
    assert 'shapiro' in results
    assert 'anderson' in results
    assert 'normaltest' in results
    assert 'lilliefors' in results


def test_normality_specific_method_shapiro(sample_normality_df):
    """Test the functionality of the Shapiro-Wilk normality test."""
    results = evaluate_normality(sample_normality_df, 'NumericData', 'Group', method='shapiro', pipeline=False)
    assert isinstance(results, dict)
    assert all(['stat' in result and 'p' in result for result in results['shapiro'].values()])


def test_normality_specific_method_anderson(sample_normality_df):
    """Test the functionality of the Anderson-Darling normality test."""
    results = evaluate_normality(sample_normality_df, 'NumericData', 'Group', method='anderson', pipeline=False)
    assert isinstance(results, dict)
    assert all(['stat' in result and 'p' in result for result in results['anderson'].values()])


def test_normality_specific_method_normaltest(sample_normality_df):
    """Test the functionality of D'Agostino's K^2 normality test."""
    results = evaluate_normality(sample_normality_df, 'NumericData', 'Group', method='normaltest', pipeline=False)
    assert isinstance(results, dict)
    assert all(['stat' in result and 'p' in result for result in results['normaltest'].values()])


def test_normality_specific_method_lilliefors(sample_normality_df):
    """Test the functionality of Lilliefors normality test."""
    results = evaluate_normality(sample_normality_df, 'NumericData', 'Group', method='lilliefors', pipeline=False)
    assert isinstance(results, dict)
    assert all(['stat' in result and 'p' in result for result in results['lilliefors'].values()])


def test_normality_pipeline_true(sample_normality_df):
    """Test the pipeline mode returns only a boolean indicating normality based on consensus method."""
    result = evaluate_normality(sample_normality_df, 'NumericData', 'Group', pipeline=True)
    assert isinstance(result, bool)


def test_grouping_by_categorical_variable(sample_normality_df):
    """Test normality tests are correctly applied to different groups defined by the categorical variable."""
    results = evaluate_normality(sample_normality_df, 'NumericData', 'Group', method='consensus', pipeline=False)
    groups = sample_normality_df['Group'].unique()
    # Ensure results are returned for each group
    assert all(group in results['shapiro'] for group in groups)
    assert all(group in results['anderson'] for group in groups)
    assert all(group in results['normaltest'] for group in groups)
    assert all(group in results['lilliefors'] for group in groups)


def test_method_output_differences(sample_normality_df):
    """Test differences in output based on method specified in non-pipeline mode."""
    consensus_results = evaluate_normality(sample_normality_df, 'NumericData', 'Group', method='consensus', pipeline=False)
    shapiro_results = evaluate_normality(sample_normality_df, 'NumericData', 'Group', method='shapiro', pipeline=False)
    # Assert that consensus results include more information
    assert len(consensus_results) >= len(shapiro_results['shapiro'])
    # Check that the keys are appropriately named
    assert 'shapiro' in consensus_results
