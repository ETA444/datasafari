import pytest
import numpy as np
import pandas as pd
from datasafari.predictor.predict_hypothesis import hypothesis_predictor_core_n, hypothesis_predictor_core_c, predict_hypothesis


@pytest.fixture
def sample_data_hpcn():
    """ Provides sample data for hypothesis_predictor_core_n() tests. """
    return pd.DataFrame({
        'Group': np.random.choice(['A', 'B', 'C'], 100),
        'Value': np.random.normal(0, 1, 100)
    })


@pytest.fixture
def sample_contingency_table():
    """ Provides a sample contingency table for hypothesis_predictor_core_c() tests. """
    return pd.crosstab(
        index=np.random.choice(['A', 'B'], 50),
        columns=np.random.choice(['X', 'Y'], 50)
    )


# TESTING ERROR-HANDLING for hypothesis_predictor_core_n() #

def test_hypothesis_predictor_core_n_non_dataframe_input():
    """ Test that non-DataFrame input raises a TypeError. """
    with pytest.raises(TypeError, match="pandas DataFrame"):
        hypothesis_predictor_core_n("not_a_dataframe", 'Value', 'Group', True, True)


def test_hypothesis_predictor_core_n_non_string_target_variable(sample_data_hpcn):
    """ Test that non-string target_variable raises a TypeError. """
    with pytest.raises(TypeError, match="must be a string"):
        hypothesis_predictor_core_n(sample_data_hpcn, 123, 'Group', True, True)


def test_hypothesis_predictor_core_n_non_string_grouping_variable(sample_data_hpcn):
    """ Test that non-string grouping_variable raises a TypeError. """
    with pytest.raises(TypeError, match="must be a string"):
        hypothesis_predictor_core_n(sample_data_hpcn, 'Value', 123, True, True)


def test_hypothesis_predictor_core_n_non_boolean_normality_bool(sample_data_hpcn):
    """ Test that non-boolean normality_bool raises a TypeError. """
    with pytest.raises(TypeError, match="must be a boolean"):
        hypothesis_predictor_core_n(sample_data_hpcn, 'Value', 'Group', 'yes', True)


def test_hypothesis_predictor_core_n_non_boolean_equal_variances_bool(sample_data_hpcn):
    """ Test that non-boolean equal_variances_bool raises a TypeError. """
    with pytest.raises(TypeError, match="must be a boolean"):
        hypothesis_predictor_core_n(sample_data_hpcn, 'Value', 'Group', True, 'yes')


def test_hypothesis_predictor_core_n_empty_dataframe():
    """ Test that empty DataFrame raises a ValueError. """
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError, match="input DataFrame is empty"):
        hypothesis_predictor_core_n(df_empty, 'Value', 'Group', True, True)


def test_hypothesis_predictor_core_n_non_existent_target_variable(sample_data_hpcn):
    """ Test that non-existent target_variable raises a ValueError. """
    with pytest.raises(ValueError, match="target variable 'NonExistentVar'"):
        hypothesis_predictor_core_n(sample_data_hpcn, 'NonExistentVar', 'Group', True, True)


def test_hypothesis_predictor_core_n_non_existent_grouping_variable(sample_data_hpcn):
    """ Test that non-existent grouping_variable raises a ValueError. """
    with pytest.raises(ValueError, match="grouping variable 'NonExistentVar'"):
        hypothesis_predictor_core_n(sample_data_hpcn, 'Value', 'NonExistentVar', True, True)


def test_hypothesis_predictor_core_n_non_numerical_target_variable(sample_data_hpcn):
    """ Test that non-numerical target_variable raises a ValueError. """
    sample_data_hpcn['NonNumeric'] = ['A'] * 100
    with pytest.raises(ValueError, match="must be a numerical variable"):
        hypothesis_predictor_core_n(sample_data_hpcn, 'NonNumeric', 'Group', True, True)


def test_hypothesis_predictor_core_n_non_categorical_grouping_variable(sample_data_hpcn):
    """ Test that non-categorical grouping_variable raises a ValueError. """
    sample_data_hpcn['NonCategorical'] = np.linspace(0, 1, 100)
    with pytest.raises(ValueError, match="must be a categorical variable"):
        hypothesis_predictor_core_n(sample_data_hpcn, 'Value', 'NonCategorical', True, True)


# TESTING FUNCTIONALITY for hypothesis_predictor_core_n() #

def test_hypothesis_predictor_core_n_ttest(sample_data_hpcn):
    """ Test independent samples t-test for two groups. """
    sample_data_hpcn['Group'] = np.random.choice(['A', 'B'], 100)
    result = hypothesis_predictor_core_n(sample_data_hpcn, 'Value', 'Group', True, True)
    assert 'ttest_ind' in result
    assert 'stat' in result['ttest_ind']
    assert 'p_val' in result['ttest_ind']
    assert result['ttest_ind']['test_name'] == 'Independent Samples T-Test'


def test_hypothesis_predictor_core_n_mannwhitney(sample_data_hpcn):
    """ Test Mann-Whitney U test for two groups. """
    sample_data_hpcn['Group'] = np.random.choice(['A', 'B'], 100)
    result = hypothesis_predictor_core_n(sample_data_hpcn, 'Value', 'Group', False, False)
    assert 'mannwhitneyu' in result
    assert 'stat' in result['mannwhitneyu']
    assert 'p_val' in result['mannwhitneyu']
    assert result['mannwhitneyu']['test_name'] == 'Mann-Whitney U Rank Test (Two Independent Samples)'


def test_hypothesis_predictor_core_n_anova(sample_data_hpcn):
    """ Test one-way ANOVA for three groups. """
    sample_data_hpcn['Group'] = np.random.choice(['A', 'B', 'C'], 100)
    result = hypothesis_predictor_core_n(sample_data_hpcn, 'Value', 'Group', True, True)
    assert 'f_oneway' in result
    assert 'stat' in result['f_oneway']
    assert 'p_val' in result['f_oneway']
    assert result['f_oneway']['test_name'] == 'One-way ANOVA (with 3 groups)'


def test_hypothesis_predictor_core_n_kruskal(sample_data_hpcn):
    """ Test Kruskal-Wallis H-test for three groups. """
    sample_data_hpcn['Group'] = np.random.choice(['A', 'B', 'C'], 100)
    result = hypothesis_predictor_core_n(sample_data_hpcn, 'Value', 'Group', False, False)
    assert 'kruskal' in result
    assert 'stat' in result['kruskal']
    assert 'p_val' in result['kruskal']
    assert result['kruskal']['test_name'] == 'Kruskal-Wallis H-test (with 3 groups)'


def test_hypothesis_predictor_core_n_more_than_three_groups(sample_data_hpcn):
    """ Test ANOVA with more than three groups. """
    sample_data_hpcn['Group'] = np.random.choice(['A', 'B', 'C', 'D'], 100)
    result = hypothesis_predictor_core_n(sample_data_hpcn, 'Value', 'Group', True, True)
    assert 'f_oneway' in result
    assert 'stat' in result['f_oneway']
    assert 'p_val' in result['f_oneway']
    assert result['f_oneway']['test_name'] == 'One-way ANOVA (with 4 groups)'


# TESTING ERROR-HANDLING for hypothesis_predictor_core_c() #

def test_hypothesis_predictor_core_c_invalid_df():
    """ Test that non-DataFrame input raises a TypeError. """
    with pytest.raises(TypeError):
        hypothesis_predictor_core_c("not a dataframe", True, True, True, True, True)


def test_hypothesis_predictor_core_c_invalid_chi2_viability(sample_contingency_table):
    """ Test that non-boolean chi2_viability raises a TypeError. """
    with pytest.raises(TypeError):
        hypothesis_predictor_core_c(sample_contingency_table, "not boolean", True, True, True, True)


def test_hypothesis_predictor_core_c_invalid_barnard_viability(sample_contingency_table):
    """ Test that non-boolean barnard_viability raises a TypeError. """
    with pytest.raises(TypeError):
        hypothesis_predictor_core_c(sample_contingency_table, True, "not boolean", True, True, True)


def test_hypothesis_predictor_core_c_invalid_boschloo_viability(sample_contingency_table):
    """ Test that non-boolean boschloo_viability raises a TypeError. """
    with pytest.raises(TypeError):
        hypothesis_predictor_core_c(sample_contingency_table, True, True, "not boolean", True, True)


def test_hypothesis_predictor_core_c_invalid_fisher_viability(sample_contingency_table):
    """ Test that non-boolean fisher_viability raises a TypeError. """
    with pytest.raises(TypeError):
        hypothesis_predictor_core_c(sample_contingency_table, True, True, True, "not boolean", True)


def test_hypothesis_predictor_core_c_invalid_yates_correction_viability(sample_contingency_table):
    """ Test that non-boolean yates_correction_viability raises a TypeError. """
    with pytest.raises(TypeError):
        hypothesis_predictor_core_c(sample_contingency_table, True, True, True, True, "not boolean")


def test_hypothesis_predictor_core_c_invalid_alternative_type(sample_contingency_table):
    """ Test that non-string alternative raises a TypeError. """
    with pytest.raises(TypeError):
        hypothesis_predictor_core_c(sample_contingency_table, True, True, True, True, True, alternative=123)


def test_hypothesis_predictor_core_c_invalid_alternative_value(sample_contingency_table):
    """ Test that invalid string alternative raises a ValueError. """
    with pytest.raises(ValueError):
        hypothesis_predictor_core_c(sample_contingency_table, True, True, True, True, True, alternative="invalid")


def test_hypothesis_predictor_core_c_empty_contingency_table():
    """ Test handling of empty contingency table. """
    empty_contingency_table = pd.DataFrame()
    with pytest.raises(ValueError):
        hypothesis_predictor_core_c(empty_contingency_table, True, True, True, True, True)


# TESTING FUNCTIONALITY for hypothesis_predictor_core_c() #

def test_hypothesis_predictor_core_c_chi2_without_yates(sample_contingency_table):
    """ Test Chi-square test without Yates' correction. """
    result = hypothesis_predictor_core_c(sample_contingency_table, True, False, False, False, False)
    assert "chi2_contingency" in result
    assert "stat" in result["chi2_contingency"]
    assert "p_val" in result["chi2_contingency"]
    assert "conclusion" in result["chi2_contingency"]
    assert "yates_correction" in result["chi2_contingency"]
    assert result["chi2_contingency"]["yates_correction"] is False


def test_hypothesis_predictor_core_c_chi2_with_yates(sample_contingency_table):
    """ Test Chi-square test with Yates' correction. """
    result = hypothesis_predictor_core_c(sample_contingency_table, True, False, False, False, True)
    assert "chi2_contingency" in result
    assert "stat" in result["chi2_contingency"]
    assert "p_val" in result["chi2_contingency"]
    assert "conclusion" in result["chi2_contingency"]
    assert "yates_correction" in result["chi2_contingency"]
    assert result["chi2_contingency"]["yates_correction"] is True


def test_hypothesis_predictor_core_c_barnard(sample_contingency_table):
    """ Test Barnard's exact test. """
    result = hypothesis_predictor_core_c(sample_contingency_table, False, True, False, False, False)
    assert "barnard_exact" in result
    assert "stat" in result["barnard_exact"]
    assert "p_val" in result["barnard_exact"]
    assert "conclusion" in result["barnard_exact"]
    assert "alternative" in result["barnard_exact"]
    assert result["barnard_exact"]["alternative"] == "two-sided"


def test_hypothesis_predictor_core_c_boschloo(sample_contingency_table):
    """ Test Boschloo's exact test. """
    result = hypothesis_predictor_core_c(sample_contingency_table, False, False, True, False, False)
    assert "boschloo_exact" in result
    assert "stat" in result["boschloo_exact"]
    assert "p_val" in result["boschloo_exact"]
    assert "conclusion" in result["boschloo_exact"]
    assert "alternative" in result["boschloo_exact"]
    assert result["boschloo_exact"]["alternative"] == "two-sided"


def test_hypothesis_predictor_core_c_fisher(sample_contingency_table):
    """ Test Fisher's exact test. """
    result = hypothesis_predictor_core_c(sample_contingency_table, False, False, False, True, False)
    assert "fisher_exact" in result
    assert "stat" in result["fisher_exact"]
    assert "p_val" in result["fisher_exact"]
    assert "conclusion" in result["fisher_exact"]
    assert "alternative" in result["fisher_exact"]
    assert result["fisher_exact"]["alternative"] == "two-sided"


def test_hypothesis_predictor_core_c_all_tests(sample_contingency_table):
    """ Test running all tests. """
    result = hypothesis_predictor_core_c(sample_contingency_table, True, True, True, True, True)
    assert "chi2_contingency" in result or "barnard_exact" in result or "boschloo_exact" in result or "fisher_exact" in result
