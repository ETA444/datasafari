import numpy as np
import pandas as pd
from scipy.stats.contingency import expected_freq
from scipy.stats import (
    ttest_ind, mannwhitneyu, f_oneway, kruskal,  # numerical hypothesis testing
    chi2_contingency, barnard_exact, boschloo_exact, fisher_exact  # categorical hypothesis testing
)
from datasafari.evaluator.evaluate_dtype import evaluate_dtype
from datasafari.evaluator.evaluate_variance import evaluate_variance
from datasafari.evaluator.evaluate_normality import evaluate_normality
from datasafari.evaluator.evaluate_contingency_table import evaluate_contingency_table


# Hypothesis Predictor Cores
def hypothesis_predictor_core_n(
        df: pd.DataFrame,
        target_variable: str,
        grouping_variable: str,
        normality_bool: bool,
        equal_variances_bool: bool
) -> dict:
    """
    **Conducts hypothesis testing on numerical data, choosing appropriate tests based on data characteristics.**

    This function performs hypothesis testing between groups defined by a categorical variable for a numerical target variable. It selects the appropriate statistical test based on the normality of the data and the homogeneity of variances across groups, utilizing t-tests, Mann-Whitney U tests, ANOVA, or Kruskal-Wallis tests as appropriate.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to be analyzed.
    target_variable : str
        The name of the numerical variable to test across groups.
    grouping_variable : str
        The name of the categorical variable used to define groups.
    normality_bool : bool
        A boolean indicating if the data follows a normal distribution within groups.
    equal_variances_bool : bool
        A boolean indicating if the groups have equal variances.

    Returns:
    --------
    dict
        A dictionary containing the results of the hypothesis test, including test statistics, p-values, conclusions regarding the differences between groups, the name of the test used, and the assumptions tested (normality and equal variances).

    Raises:
    -------
    TypeErrors:
        - If `df` is not a pandas DataFrame.
        - If `target_variable` or `grouping_variable` is not a string.
        - If `normality_bool` or `equal_variances_bool` is not a boolean.
    ValueErrors:
        - If the `df` is empty, indicating that there's no data to evaluate.
        - If `target_variable` or `grouping_variable` is not found in the DataFrame's columns.
        - If `target_variable` is not numerical.
        - If `grouping_variable` is not categorical.
    """

    # Error Handling
    # TypeErrors
    if not isinstance(df, pd.DataFrame):
        raise TypeError("predictor_core_numerical(): The 'df' parameter must be a pandas DataFrame.")

    if not isinstance(target_variable, str):
        raise TypeError("predictor_core_numerical(): The 'target_variable' must be a string.")

    if not isinstance(grouping_variable, str):
        raise TypeError("predictor_core_numerical(): The 'grouping_variable' must be a string.")

    if not isinstance(normality_bool, bool):
        raise TypeError("predictor_core_numerical(): The 'normality_bool' must be a boolean.")

    if not isinstance(equal_variances_bool, bool):
        raise TypeError("predictor_core_numerical(): The 'equal_variances_bool' must be a boolean.")

    # ValueErrors
    if df.empty:
        raise ValueError("predictor_core_n(): The input DataFrame is empty.")

    if target_variable not in df.columns:
        raise ValueError(f"predictor_core_n(): The target variable '{target_variable}' was not found in the DataFrame.")

    if grouping_variable not in df.columns:
        raise ValueError(f"predictor_core_n(): The grouping variable '{grouping_variable}' was not found in the DataFrame.")

    target_variable_is_numerical = evaluate_dtype(df, [target_variable], output='list_n')[0]
    if not target_variable_is_numerical:
        raise ValueError(f"predictor_core_n(): The target variable '{target_variable}' must be a numerical variable.")

    grouping_variable_is_categorical = evaluate_dtype(df, [grouping_variable], output='list_c')[0]
    if not grouping_variable_is_categorical:
        raise ValueError(f"predictor_core_n(): The grouping variable '{grouping_variable}' must be a categorical variable.")

    # Main Function
    groups = df[grouping_variable].unique().tolist()
    samples = [df[df[grouping_variable] == group][target_variable] for group in groups]

    # define output object
    output_info = {}

    if len(samples) == 2:  # two sample testing
        if normality_bool and equal_variances_bool:  # parametric testing
            stat, p_val = ttest_ind(*samples)
            test_name = 'Independent Samples T-Test'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference between the group means, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference between the group means, rejecting the null hypothesis."
            output_info['ttest_ind'] = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion, 'test_name': test_name, 'normality': normality_bool, 'equal_variance': equal_variances_bool}
        else:  # non-parametric testing
            stat, p_val = mannwhitneyu(*samples)
            test_name = 'Mann-Whitney U Rank Test (Two Independent Samples)'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference in group distributions, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference in group distributions, rejecting the null hypothesis."
            output_info['mannwhitneyu'] = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion, 'test_name': test_name, 'normality': normality_bool, 'equal_variance': equal_variances_bool}
    else:  # more than two samples
        if normality_bool and equal_variances_bool:
            stat, p_val = f_oneway(*samples)
            test_name = f'One-way ANOVA (with {len(samples)} groups)'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference among the group means, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference among the group means, rejecting the null hypothesis."
            output_info['f_oneway'] = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion, 'test_name': test_name, 'normality': normality_bool, 'equal_variance': equal_variances_bool}
        else:
            stat, p_val = kruskal(*samples)
            test_name = f'Kruskal-Wallis H-test (with {len(samples)} groups)'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference among the group distributions, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference among the group distributions, rejecting the null hypothesis."
            output_info['kruskal'] = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion, 'test_name': test_name, 'normality': normality_bool, 'equal_variance': equal_variances_bool}

    # construct console output and return
    print(f"< HYPOTHESIS TESTING: {test_name}>\nBased on:\n  ➡ Normality assumption: {'✔' if normality_bool else '✘'}\n  ➡ Equal variances assumption: {'✔' if equal_variances_bool else '✘'}\n  ➡ Nr. of Groups: {len(samples)} groups\n  ∴ predict_hypothesis() is performing {test_name}:\n")
    print(f"Results of {test_name}:\n  ➡ statistic: {stat}\n  ➡ p-value: {p_val}\n  ∴ Conclusion: {conclusion}\n")
    return output_info


def hypothesis_predictor_core_c(
        contingency_table: pd.DataFrame,
        chi2_viability: bool,
        barnard_viability: bool,
        boschloo_viability: bool,
        fisher_viability: bool,
        yates_correction_viability: bool,
        alternative: str = 'two-sided'
) -> dict:
    """
    **Conducts categorical hypothesis testing using contingency tables and appropriate statistical tests.**

    This function assesses the association between two categorical variables by applying a series of statistical tests. It evaluates the data's suitability for different tests based on the shape of the contingency table, the minimum expected and observed frequencies, and specific methodological preferences, including Yates' correction for chi-square tests and alternatives for exact tests.

    Parameters:
    -----------
    contingency_table : pd.DataFrame
        A contingency table of the two categorical variables.
    chi2_viability : bool
        Indicates whether chi-square tests should be considered based on the data's suitability.
    barnard_viability : bool
        Indicators for the applicability of Barnard's exact test.
    boschloo_viability : bool
        Indicators for the applicability of Boschloo's exact test.
    fisher_viability : bool
        Indicators for the applicability of Fisher's exact test.
    yates_correction_viability : bool
        Determines whether Yates' correction is applicable based on the contingency table's shape and sample size.
    alternative : str, optional, default: 'two-sided'
        Specifies the alternative hypothesis for exact tests. Options include ``'two-sided'``, ``'less'``, or ``'greater'``.

    Returns:
    --------
    dict
        A comprehensive dictionary detailing the outcomes of the statistical tests performed, including test names, statistics, p-values, and conclusions about the association between the categorical variables. The dictionary also contains specific details about the application of Yates' correction and the chosen alternative hypothesis for exact tests.

    Raises:
    -------
    TypeErrors:
        - If `contingency_table` is not a pandas DataFrame.
        - If `chi2_viability`, `barnard_viability`, `boschloo_viability`, `fisher_viability`, or `yates_correction_viability` is not a boolean.
        - If `alternative` is not a string indicating the alternative hypothesis ('two-sided', 'less', 'greater').
    ValueErrors:
        - If the `contingency_table` is empty.
        - If the `alternative` specified does not match one of the expected values.
    """

    # Error Handling
    # TypeErrors
    if not isinstance(contingency_table, pd.DataFrame):
        raise TypeError("predictor_core_c(): The 'contingency_table' parameter must be a pandas DataFrame.")

    if not isinstance(chi2_viability, bool):
        raise TypeError("predictor_core_c(): The 'chi2_viability' parameter must be a boolean.")

    if not isinstance(barnard_viability, bool):
        raise TypeError("predictor_core_c(): The 'barnard_viability' parameter must be a boolean.")

    if not isinstance(boschloo_viability, bool):
        raise TypeError("predictor_core_c(): The 'boschloo_viability' parameter must be a boolean.")

    if not isinstance(fisher_viability, bool):
        raise TypeError("predictor_core_c(): The 'fisher_viability' parameter must be a boolean.")

    if not isinstance(yates_correction_viability, bool):
        raise TypeError("predictor_core_c(): The 'yates_correction_viability' parameter must be a boolean.")

    if not isinstance(alternative, str):
        raise TypeError("predictor_core_c(): The 'alternative' parameter must be a string.")

    # ValueErrors
    if contingency_table.empty:
        raise ValueError("predictor_core_c(): The input contingency table is empty.")

    valid_alternatives = ['two-sided', 'less', 'greater']
    if alternative not in valid_alternatives:
        raise ValueError(f"predictor_core_c(): Invalid 'alternative' value. Expected one of {valid_alternatives}, got '{alternative}'.")

    # Main Function
    # to avoid unnecessary parameter inputs use contingency_table object
    categorical_variable1 = contingency_table.index.name
    categorical_variable2 = contingency_table.columns.name
    min_expected_frequency = expected_freq(contingency_table).min()
    min_observed_frequency = contingency_table.min().min()
    contingency_table_shape = contingency_table.shape
    sample_size = np.sum(contingency_table.values)

    # define output object
    output_info = {}

    if chi2_viability:
        if yates_correction_viability:
            stat, p_val, dof, expected_frequencies = chi2_contingency(contingency_table, correction=True)
            test_name = "Chi-square test (with Yates' Correction)"
            chi2_tip = "\n\n☻ Tip: The Chi-square test of independence with Yates' Correction is used for 2x2 contingency tables with small sample sizes. Yates' Correction makes the test more conservative, reducing the Type I error rate by adjusting for the continuity of the chi-squared distribution. This correction is typically applied when sample sizes are small (often suggested for total sample sizes less than about 40), aiming to avoid overestimation of statistical significance."
            conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f})." if p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f})."
            chi2_output_info = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion, 'yates_correction': yates_correction_viability, 'tip': chi2_tip, 'test_name': test_name}
            output_info['chi2_contingency'] = chi2_output_info
        else:
            stat, p_val, dof, expected_frequencies = chi2_contingency(contingency_table, correction=False)
            test_name = "Chi-square test (without Yates' Correction)"
            chi2_tip = "\n\n☻ Tip: The Chi-square test of independence without Yates' Correction is preferred when analyzing larger contingency tables or when sample sizes are sufficiently large, even for 2x2 tables (often suggested for total sample sizes greater than 40). Removing Yates' Correction can increase the test's power by not artificially adjusting for continuity, making it more sensitive to detect genuine associations between variables in settings where the assumptions of the chi-squared test are met."
            conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f})." if p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f})."
            chi2_output_info = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion, 'yates_correction': yates_correction_viability, 'tip': chi2_tip, 'test_name': test_name}
            output_info['chi2_contingency'] = chi2_output_info
    else:
        if barnard_viability:
            barnard_test = barnard_exact(contingency_table, alternative=alternative.lower())
            barnard_stat = barnard_test.statistic
            barnard_p_val = barnard_test.pvalue
            bernard_test_name = f"Barnard's exact test ({alternative.lower()})"
            bernard_tip = "\n\n☻ Tip: Barnard's exact test is often preferred for its power, especially in unbalanced designs or with small sample sizes, without the need for the continuity correction that Fisher's test applies."
            if alternative.lower() == 'two-sided':
                bernard_conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {barnard_p_val:.3f})." if barnard_p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {barnard_p_val:.3f})."
            elif alternative.lower() == 'less':
                bernard_conclusion = f"The data do not support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {barnard_p_val:.3f})." if barnard_p_val > 0.05 else f"The data support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {barnard_p_val:.3f})."
            elif alternative.lower() == 'greater':
                bernard_conclusion = f"The data do not support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {barnard_p_val:.3f})." if barnard_p_val > 0.05 else f"The data support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {barnard_p_val:.3f})."

            # consolidate info for output
            bernard_output_info = {'stat': barnard_stat, 'p_val': barnard_p_val, 'conclusion': bernard_conclusion, 'alternative': alternative.lower(), 'tip': bernard_tip, 'test_name': bernard_test_name}
            output_info['barnard_exact'] = bernard_output_info

        if boschloo_viability:
            boschloo_test = boschloo_exact(contingency_table, alternative=alternative.lower())
            boschloo_stat = boschloo_test.statistic
            boschloo_p_val = boschloo_test.pvalue
            boschloo_test_name = f"Boschloo's exact test ({alternative.lower()})"
            boschloo_tip = "\n\n☻ Tip: Boschloo's exact test is an extension that increases power by combining the strengths of Fisher's and Barnard's tests, focusing on the most extreme probabilities."
            if alternative.lower() == 'two-sided':
                boschloo_conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {boschloo_p_val:.3f})." if boschloo_p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {boschloo_p_val:.3f})."
            elif alternative.lower() == 'less':
                boschloo_conclusion = f"The data do not support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})." if boschloo_p_val > 0.05 else f"The data support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})."
            elif alternative.lower() == 'greater':
                boschloo_conclusion = f"The data do not support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})." if boschloo_p_val > 0.05 else f"The data support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})."

            # consolidate info for output
            boschloo_output_info = {'stat': boschloo_stat, 'p_val': boschloo_p_val, 'conclusion': boschloo_conclusion, 'alternative': alternative.lower(), 'tip': boschloo_tip, 'test_name': boschloo_test_name}
            output_info['boschloo_exact'] = boschloo_output_info

        if fisher_viability:
            fisher_test = fisher_exact(contingency_table, alternative=alternative.lower())
            fisher_stat = fisher_test[0]
            fisher_p_val = fisher_test[1]
            fisher_test_name = f"Fisher's exact test ({alternative.lower()})"
            fisher_tip = "\n☻ Tip: Fisher's exact test is traditionally used for small sample sizes, providing exact p-values under the null hypothesis of independence."
            if alternative.lower() == 'two-sided':
                fisher_conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {fisher_p_val:.3f})." if fisher_p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {fisher_p_val:.3f})."
            elif alternative.lower() == 'less':
                fisher_conclusion = f"The data do not support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {fisher_p_val:.3f})." if fisher_p_val > 0.05 else f"The data support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {fisher_p_val:.3f})."
            elif alternative.lower() == 'greater':
                fisher_conclusion = f"The data do not support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {fisher_p_val:.3f})." if fisher_p_val > 0.05 else f"The data support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {fisher_p_val:.3f})."

            # consolidate info for output
            fisher_output_info = {'stat': fisher_stat, 'p_val': fisher_p_val, 'conclusion': fisher_conclusion, 'alternative': alternative.lower(), 'tip': fisher_tip, 'test_name': fisher_test_name}
            output_info['fisher_exact'] = fisher_output_info

    # console output & return
    generate_test_names = [info['test_name'] for info in output_info.values()]
    # generate_test_names = (output_info[0]['test_name'] if len(output_info) == 1 else [info['test_name'] for info in output_info.values()])
    print(f"< HYPOTHESIS TESTING: {generate_test_names}>\nBased on:\n  ➡ Sample size: {sample_size}\n  ➡ Minimum Observed Frequency: {min_observed_frequency}\n  ➡ Minimum Expected Frequency: {min_expected_frequency}\n  ➡ Contingency table shape: {contingency_table_shape[0]}x{contingency_table_shape[1]}\n  ∴ Performing {generate_test_names}:")
    print("\n☻ Tip: Consider the tips provided for each test and assess which of the exact tests provided is most suitable for your data.") if len(output_info) > 1 else print('')
    [print(f"Results of {info['test_name']}:\n  ➡ statistic: {info['stat']}\n  ➡ p-value: {info['p_val']}\n  ∴ Conclusion: {info['conclusion']}{info['tip']}\n") for info in output_info.values()]
    return output_info


# Main function
def predict_hypothesis(
        df: pd.DataFrame,
        var1: str,
        var2: str,
        normality_method: str = 'consensus',
        variance_method: str = 'consensus',
        exact_tests_alternative: str = 'two-sided',
        yates_min_sample_size: int = 40
) -> dict:
    """
    **Conduct the optimal hypothesis test on a DataFrame, tailoring the approach based on the variable types and automating the testing prerequisites and analyses, outputting test results and interpretation.**

    This function simplifies hypothesis testing to requiring only two variables and a DataFrame, intelligently determining the test type, assessing necessary assumptions, and providing detailed test outcomes and conclusions.

        The combination of `var1` and `var2` data types determines the type of hypothesis test to perform:
            - A pairing with two categorical variables triggers categorical testing (e.g., Chi-square, Fisher's exact test, etc.).
            - A numerical and categorical variable pairing, interpreted as target and grouping variables respectively, leads to numerical testing (e.g., t-tests, ANOVA, etc.).
            - Numerical and numerical variable pairings are not supported.


    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data for hypothesis testing.

    var1 : str
        The name of the first variable for hypothesis testing.

    var2 : str
        The name of the second variable for hypothesis testing.

    normality_method : str, optional, default: 'consensus'
        Specifies the method to evaluate normality within numerical hypothesis testing.

        - ``'shapiro'`` Shapiro-Wilk test.
        - ``'anderson'`` Anderson-Darling test.
        - ``'normaltest'`` D’Agostino and Pearson’s test.
        - ``'lilliefors'`` Lilliefors test for normality.
        - ``'consensus'`` Utilizes a combination of the above tests to reach a consensus on normality.

            *Note: For more details, refer to:* :doc:`evaluate_normality() documentation <datasafari.evaluator.evaluate_normality>`

    variance_method : str, optional, default: 'consensus'
        Determines the method to evaluate variance homogeneity (equal variances) across groups in numerical hypothesis testing.

        - ``'levene'`` Levene's test, robust to non-normal distributions.
        - ``'bartlett'`` Bartlett’s test, sensitive to non-normal distributions.
        - ``'fligner'`` Fligner-Killeen test, a non-parametric alternative.
        - ``'consensus'`` A combination approach to determine equal variances across methods.

            *Note: For more details, refer to:* :doc:`evaluate_variance() documentation <datasafari.evaluator.evaluate_variance>`

    exact_tests_alternative : str, optional, default: 'two-sided'
        For categorical hypothesis testing, this parameter specifies the alternative hypothesis direction for exact tests.

        - ``'two-sided'`` Tests for any difference between the two variables without directionality.
        - ``'less'`` Tests if the first variable is less than the second variable.
        - ``'greater'`` Tests if the first variable is greater than the second variable.

    yates_min_sample_size : int, optional, default: 40
        Specifies the minimum sample size threshold for applying Yates' correction in chi-square testing to adjust for continuity. The correction is applied to 2x2 contingency tables with small sample sizes to prevent overestimation of the significance level.

    Returns:
    --------
    dict
        A dictionary with *key* the short test name (e.g. ``'f_oneway'``) and *value* another dictionary which contains all results from that test, namely:
            - ``'stat'`` The test statistic value, quantifying the degree to which the observed data conform to the null hypothesis.
            - ``'p_val'`` The p-value, indicating the probability of observing the test results under the null hypothesis.
            - ``'conclusion'`` A textual interpretation of the test outcome, stating whether the evidence was sufficient to reject the null hypothesis.
            - ``'test_name'`` The full name of the statistical test performed (e.g., 'Independent Samples T-Test', 'Chi-square test').

                *Additional values in certain scenarios may be:*
                    - ``'alternative'`` Specifies the alternative hypothesis direction used in exact tests ('two-sided', 'less', 'greater').
                    - ``'yates_correction'`` A boolean that indicates whether a Yate's correction was applied used in Chi-square test.
                    - ``'normality'`` A boolean that indicates whether the data were found to meet the normality assumption.
                    - ``'equal_variance'`` A boolean that indicates whether the data were found to have equal variances across groups.
                    - ``'tip'`` Helpful insights or considerations regarding the test's application or interpretation.

    Raises:
    -------
    TypeErrors:
        - If `df` is not a pandas DataFrame.
        - If `var1` or `var2` is not a string.
        - If `normality_method`, `variance_method`, or `exact_tests_alternative` is not a string.
        - If `yates_min_sample_size` is not an integer.
    ValueErrors:
        - If the `df` is empty.
        - If `normality_method` is not one of the valid options.
        - If `variance_method` is not one of the valid options.
        - If `exact_tests_alternative` is not one of the valid options.
        - If `yates_min_sample_size` is less than 1.


    Examples:
    ---------
    First, we create a DataFrame with categorical and numerical variables to use in our examples:

    >>> import datasafari
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'Group': np.random.choice(['Control', 'Treatment'], size=100),
    ...     'Score': np.random.normal(0, 1, 100),
    ...     'Category': np.random.choice(['Type1', 'Type2'], size=100),
    ...     'Feature2': np.random.exponential(1, 100)
    ... })

    Scenario 1: Basic numerical hypothesis testing (T-test or ANOVA based on groups)

    >>> output_num_basic = predict_hypothesis(df, 'Group', 'Score')

    Scenario 2: Numerical hypothesis testing specifying method to evaluate normality

    >>> output_num_normality = predict_hypothesis(df, 'Group', 'Score', normality_method='shapiro')

    Scenario 3: Numerical hypothesis testing with a specified method to evaluate variance

    >>> output_num_variance = predict_hypothesis(df, 'Group', 'Score', variance_method='levene')

    Scenario 4: Categorical hypothesis testing (Chi-square or Fisher's exact test)

    >>> output_cat_basic = predict_hypothesis(df, 'Group', 'Category')

    Scenario 5: Categorical hypothesis testing with alternative hypothesis specified

    >>> output_cat_alternative = predict_hypothesis(df, 'Category', 'Group', exact_tests_alternative='less')

    Scenario 6: Applying Yates' correction in a Chi-square test for small samples

    >>> output_yates_correction = predict_hypothesis(df, 'Group', 'Category', yates_min_sample_size=30)

    Scenario 7: Comprehensive numerical hypothesis testing using consensus for normality and variance evaluation

    >>> output_num_comprehensive = predict_hypothesis(df, 'Group', 'Score', normality_method='consensus', variance_method='consensus')

    Scenario 8: Testing with a numerical variable against a different grouping variable

    >>> output_different_group = predict_hypothesis(df, 'Feature2', 'Group')

    Scenario 9: Exploring exact tests in categorical hypothesis testing for a 2x2 table

    >>> df_small = df.sample(20) # Smaller sample for demonstration
    >>> output_exact_tests = predict_hypothesis(df_small, 'Category', 'Group', exact_tests_alternative='two-sided')

    Notes:
    ------
    ``predict_hypothesis()`` is engineered to facilitate an intuitive yet powerful entry into hypothesis testing.

    Here is a deeper look into its operational logic:
        1. **Type Determination and Variable Interpretation**:

            - **Numerical Testing**: Activated when one variable is numerical and the other categorical. The numerical variable is considered the 'target variable', subject to hypothesis testing across groups defined by the categorical 'grouping variable'.
            - **Categorical Testing**: Engaged when both variables are categorical, examining the association between them through appropriate exact tests.

        2. **Assumption Evaluation and Preparatory Checks**:

            - For **numerical data**, it evaluates:
                - **Normality**: Using methods such as Shapiro-Wilk, Anderson-Darling, D'Agostino's K-squared test, and Lilliefors test to assess the distribution of data.
                - **Homogeneity of Variances**: With Levene, Bartlett, or Fligner-Killeen tests to ensure variance uniformity across groups, guiding the choice between parametric and non-parametric tests.

            - For **categorical data**, it checks:
                - **Adequacy of Frequencies**: Ensuring observed and expected frequencies support the validity of Chi-square and other exact tests.
                - **Table Shape**: Determining the applicability of tests like Fisher’s exact test or Barnard’s test, based on the contingency table's dimensions.

        3. **Test Selection and Execution**:

            - **Numerical Hypothesis Tests** may include:
                - T-tests (independent samples, paired samples) for normally distributed data with equal variances.
                - ANOVA or Welch's ANOVA for comparing more than two groups, under respective assumptions.
                - Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis H, or Friedman tests as non-parametric alternatives.

            - **Categorical Hypothesis Tests** encompass:
                - Chi-square test of independence, with or without Yates’ correction, for general association between two categorical variables.
                - Fisher’s exact test for small sample sizes or when Chi-square assumptions are not met.
                - Barnard’s exact test, offering more power in some scenarios compared to Fisher’s test.
                - Boschloo’s exact test, aiming to increase the power further by combining strengths of Fisher’s and Barnard’s tests.

        4. **Conclusive Results and Interpretation**: Outputs include test statistics, p-values, and clear conclusions.

            - The function demystifies statistical analysis, making it approachable for users across various disciplines, enabling informed decisions based on robust statistical evidence.

    This function stands out by automating complex decision trees involved in statistical testing, offering a simplified yet comprehensive approach to hypothesis testing. It exemplifies how advanced statistical analysis can be made accessible and actionable, fostering data-driven decision-making.
    """
    # Error Handling
    # TypeErrors
    if not isinstance(df, pd.DataFrame):
        raise TypeError("predict_hypothesis(): The 'df' parameter must be a pandas DataFrame.")

    if not isinstance(var1, str) or not isinstance(var2, str):
        raise TypeError("predict_hypothesis(): The 'var1' and 'var2' parameters must be strings.")

    if not isinstance(normality_method, str):
        raise TypeError("predict_hypothesis(): The 'normality_method' parameter must be a string.")

    if not isinstance(variance_method, str):
        raise TypeError("predict_hypothesis(): The 'variance_method' parameter must be a string.")

    if not isinstance(exact_tests_alternative, str):
        raise TypeError("predict_hypothesis(): The 'exact_tests_alternative' parameter must be a string.")

    if not isinstance(yates_min_sample_size, int):
        raise TypeError("predict_hypothesis(): The 'yates_min_sample_size' parameter must be an integer.")

    # ValueErrors
    if df.empty:
        raise ValueError("model_recommendation_core_inference(): The input DataFrame is empty.")

    valid_normality_methods = ['shapiro', 'anderson', 'normaltest', 'lilliefors', 'consensus']
    if normality_method.lower() not in valid_normality_methods:
        raise ValueError(f"predict_hypothesis(): Invalid 'normality_method' value. Expected one of {valid_normality_methods}, got '{normality_method}'.")

    valid_variance_methods = ['levene', 'bartlett', 'fligner', 'consensus']
    if variance_method.lower() not in valid_variance_methods:
        raise ValueError(f"predict_hypothesis(): Invalid 'variance_method' value. Expected one of {valid_variance_methods}, got '{variance_method}'.")

    valid_alternatives = ['two-sided', 'less', 'greater']
    if exact_tests_alternative.lower() not in valid_alternatives:
        raise ValueError(f"predict_hypothesis(): Invalid 'exact_tests_alternative' value. Expected one of {valid_alternatives}, got '{exact_tests_alternative}'.")

    if yates_min_sample_size < 1:
        raise ValueError("predict_hypothesis(): The 'yates_min_sample_size' must be at least 1.")

    # Main Function
    # determine appropriate testing procedure and variable interpretation
    data_types = evaluate_dtype(df, col_names=[var1, var2], output='dict')

    if data_types[var1] == 'categorical' and data_types[var2] == 'numerical':
        hypothesis_testing = 'numerical'
        grouping_variable = var1
        target_variable = var2
        print("< INITIALIZING predict_hypothesis() >\n")
        print(f"Performing {hypothesis_testing} hypothesis testing with:")
        print(f"  ➡ Grouping variable: '{grouping_variable}' (with groups: {df[grouping_variable].unique()})\n  ➡ Target variable: '{target_variable}'\n")
        print("Output Contents:\n (1) Results of Normality Testing\n (2) Results of Variance Testing\n (3) Results of Hypothesis Testing\n\n")
    elif data_types[var1] == 'numerical' and data_types[var2] == 'categorical':
        hypothesis_testing = 'numerical'
        grouping_variable = var2
        target_variable = var1
        print("< INITIALIZING predict_hypothesis() >\n")
        print(f"Performing {hypothesis_testing} hypothesis testing with:")
        print(f"  ➡ Grouping variable: '{grouping_variable}' (with groups: {df[grouping_variable].unique()})\n  ➡ Target variable: '{target_variable}'\n")
        print("Output Contents:\n (1) Results of Normality Testing\n (2) Results of Variance Testing\n (3) Results of Hypothesis Testing\n\n")
    elif data_types[var1] == 'categorical' and data_types[var2] == 'categorical':
        hypothesis_testing = 'categorical'
        categorical_variable1 = var1
        categorical_variable2 = var2
        print("< INITIALIZING predict_hypothesis() >")
        print(f"Performing {hypothesis_testing} hypothesis testing with:\n")
        print(f"  ➡ Categorical variable 1: '{categorical_variable1}'\n  ➡ Categorical variable 2: '{categorical_variable2}'\n\n")
    else:
        raise ValueError("predict_hypothesis(): Both of the provided variables are numerical.\n - To do numerical hypothesis testing, provide a numerical variable (target variable) and a categorical variable (grouping variable).\n - To do categorical hypothesis testing, provide two categorical variables.")

    # perform appropriate hypothesis testing process
    if hypothesis_testing == 'numerical':
        # evaluate test assumptions
        normality_bool = evaluate_normality(df, target_variable, grouping_variable, method=normality_method.lower(), pipeline=True)
        equal_variance_bool = evaluate_variance(df, target_variable, grouping_variable, normality_info=normality_bool, method=variance_method.lower(), pipeline=True)

        # perform hypothesis testing
        output_info = hypothesis_predictor_core_n(df, target_variable, grouping_variable, normality_bool, equal_variance_bool)
        return output_info
    elif hypothesis_testing == 'categorical':
        # create contingency table
        contingency_table = pd.crosstab(df[categorical_variable1], df[categorical_variable2])

        # evaluate test criteria
        chi2_viability, yates_correction_viability, barnard_viability, boschloo_viability, fisher_viability = evaluate_contingency_table(contingency_table, min_sample_size_yates=yates_min_sample_size, pipeline=True, quiet=True)

        # perform hypothesis testing
        output_info = hypothesis_predictor_core_c(contingency_table, chi2_viability, barnard_viability, boschloo_viability, fisher_viability, yates_correction_viability, alternative=exact_tests_alternative.lower())
        return output_info
