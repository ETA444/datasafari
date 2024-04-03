import numpy as np
import pandas as pd
from scipy.stats import (
    ttest_ind, mannwhitneyu, f_oneway, kruskal,  # numerical hypothesis testing
    chi2_contingency, barnard_exact, boschloo_exact, fisher_exact  # categorical hypothesis testing
)
from scipy.stats.contingency import expected_freq
from datasafari.evaluator import evaluate_normality, evaluate_variance, evaluate_dtype


# Utility functions: used only in this module
def evaluate_frequencies(df: pd.DataFrame, categorical_variable1: str, categorical_variable2: str):
    """
    Evaluates the adequacy of observed and expected frequencies for applying the chi-square test of independence.

    This function calculates a contingency table from two categorical variables and checks if the data meets
    the minimum expected frequency criteria for the chi-square test of independence. It ensures that the chi-square test
    results will be reliable by verifying that both observed and expected frequencies in the contingency table
    are sufficient (typically >= 5).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the categorical data to be evaluated.
    categorical_variable1 : str
        The name of the first categorical variable for constructing the contingency table.
    categorical_variable2 : str
        The name of the second categorical variable for constructing the contingency table.

    Returns
    -------
    chi2_bool : bool
        Boolean indicating if the data meets the minimum frequency criteria for chi-square testing.
    contingency_table : pd.DataFrame
        The generated contingency table used for the chi-square test.

    Examples
    --------
    # Example usage
    >>> import pandas as pd
    >>> import numpy as np
    >>> df_example = pd.DataFrame({
    ...     'Gender': np.random.choice(['Male', 'Female'], 100),
    ...     'Preference': np.random.choice(['Option A', 'Option B', 'Option C'], 100)
    ... })
    >>> chi2_bool, contingency_table = evaluate_frequencies(df_example, 'Gender', 'Preference')
    """

    # initialize contingency table
    contingency_table = pd.crosstab(df[categorical_variable1], df[categorical_variable2])
    # compute minimum expected and observed frequencies
    min_expected_frequency = expected_freq(contingency_table).min()
    min_observed_frequency = contingency_table.min().min()
    # test assumption of chi2_contingency
    chi2_bool = False
    if min_expected_frequency >= 5 and min_observed_frequency >= 5:
        chi2_bool = True
    return chi2_bool, contingency_table


def evaluate_shape(contingency_table: pd.DataFrame):
    """
    Determines the suitability of a contingency table's shape for various statistical tests.

    This function evaluates the shape of a provided contingency table to determine which statistical tests
    can be appropriately applied. Specifically, it checks if the table is 2x2, which enables the use of
    Barnard's exact test, Boschloo's exact test, Fisher's exact test, and potentially applies Yates' correction
    for the chi-square test.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        A contingency table generated from two categorical variables.

    Returns
    -------
    barnard_bool : bool
        Indicates if Barnard's exact test can be applied (True if table is 2x2).
    boschloo_bool : bool
        Indicates if Boschloo's exact test can be applied (True if table is 2x2).
    fisher_bool : bool
        Indicates if Fisher's exact test can be applied (True if table is 2x2).
    yates_correction_shape_bool : bool
        Indicates if Yates' correction for chi-square test is applicable (True if table is 2x2).

    Examples
    --------
    # Example usage with a programmatically created contingency table
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {
    ...     'Gender': np.random.choice(['Male', 'Female'], 100),
    ...     'Preference': np.random.choice(['Option A', 'Option B'], 100)
    ... }
    >>> df_example = pd.DataFrame(data)
    >>> contingency_table = pd.crosstab(df_example['Gender'], df_example['Preference'])
    >>> barnard_bool, boschloo_bool, fisher_bool, yates_correction_shape_bool = evaluate_shape(contingency_table)
    """
    barnard_bool = False
    boschloo_bool = False
    fisher_bool = False
    yates_correction_shape_bool = False  # note: related to chi2_contingency parameter 'correction'
    if contingency_table.shape == (2,2):  # if table is 2x2 various tests become viable
        barnard_bool = True
        boschloo_bool = True
        fisher_bool = True
        yates_correction_shape_bool = True
    return barnard_bool, boschloo_bool, fisher_bool, yates_correction_shape_bool


def evaluate_contingency_table(contingency_table: pd.DataFrame, min_sample_size_yates: int = 40, pipeline: bool = False, quiet: bool = False):
    """
    Evaluates a contingency table to determine the viability of various statistical tests based on the table's characteristics.

    This function assesses the contingency table's suitability for chi-square tests, exact tests (Barnard's, Boschloo's, and Fisher's), and the application of Yates' correction within the chi-square test. It examines expected and observed frequencies, sample size, and table shape to guide the choice of appropriate statistical tests for hypothesis testing.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        A contingency table generated from two categorical variables.
    min_sample_size_yates : int, optional
        The minimum sample size below which Yates' correction should be considered. Default is 40.
    pipeline : bool, optional
        Determines the format of the output. If True, outputs a tuple of boolean values representing the viability of each test. If False, outputs a dictionary with the test names as keys and their viabilities as boolean values. Default is False.
    quiet : bool, optional

    Returns
    -------
    test_viability : dict or tuple
        Depending on the 'pipeline' parameter:
            - If `pipeline` is False, returns a dictionary with keys as test names ('chi2_contingency', 'yates_correction', 'barnard_exact', 'boschloo_exact', 'fisher_exact') and values as boolean indicators of their viability.
            - If `pipeline` is True, returns a tuple of boolean values in the order: (chi2_viability, yates_correction_viability, barnard_viability, boschloo_viability, fisher_viability).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {
    ...     'Gender': np.random.choice(['Male', 'Female'], 100),
    ...     'Preference': np.random.choice(['Option A', 'Option B'], 100)
    ... }
    >>> df_example = pd.DataFrame(data)
    >>> contingency_table = pd.crosstab(df_example['Gender'], df_example['Preference'])
    >>> test_viability = evaluate_contingency_table(contingency_table)
    >>> print(test_viability)
    >>> chi2, yates, barnard, boschloo, fisher = evaluate_contingency_table(contingency_table, pipeline=True, quiet=True)
    >>> if chi2:
    >>>     # ...
    """

    test_viability = {}  # non-pipeline output

    # compute objects for checks
    min_expected_frequency = expected_freq(contingency_table).min()
    min_observed_frequency = contingency_table.min().min()
    sample_size = np.sum(contingency_table.values)
    table_shape = contingency_table.shape

    # assumption check for chi2_contingency test
    chi2_viability = True if min_expected_frequency >= 5 and min_observed_frequency >= 5 else False
    test_viability['chi2_contingency'] = chi2_viability

    # assumption check for chi2_contingency yate's-correction
    yates_correction_viability = True if table_shape == (2, 2) and sample_size < min_sample_size_yates else False
    test_viability['yates_correction'] = yates_correction_viability

    # assumption check for all exact tests
    barnard_viability, boschloo_viability, fisher_viability = (True, True, True) if table_shape == (2, 2) else (False, False, False)
    test_viability['barnard_exact'], test_viability['boschloo_exact'], test_viability['fisher_exact'] = barnard_viability, boschloo_viability, fisher_viability

    # console output
    title = f"< CONTINGENCY TABLE EVALUATION >\n"
    on_chi2 = f"Based on minimum expected freq. ({min_expected_frequency}) & minimum observed freq. ({min_observed_frequency}):\n  ➡ chi2_contingecy() viability: {'✔' if chi2_viability else '✘'}\n\n"
    on_yates = f"Based on table shape ({table_shape[0]}x{table_shape[1]}) & sample size ({sample_size}):\n  ➡ chi2_contingecy() Yate's correction viability: {'✔' if yates_correction_viability else '✘'}\n\n"
    on_exact = f"Based on table shape ({table_shape[0]}x{table_shape[1]}):\n  ➡ barnard_exact() viability: {'✔' if barnard_viability else '✘'}\n  ➡ boschloo_exact() viability: {'✔' if boschloo_viability else '✘'}\n  ➡ fisher_exact() viability: {'✔' if fisher_viability else '✘'}\n\n\n"
    print(title, on_chi2, on_yates, on_exact) if not quiet else ""

    if pipeline:
        return chi2_viability, yates_correction_viability, barnard_viability, boschloo_viability, fisher_viability
    elif not pipeline:
        return test_viability


def predictor_core_numerical(df: pd.DataFrame, target_variable: str, grouping_variable: str, normality_bool: bool, equal_variances_bool: bool):
    """
    Conducts hypothesis testing on numerical data, choosing appropriate tests based on data characteristics.

    This function performs hypothesis testing between groups defined by a categorical variable for a numerical
    target variable. It selects the appropriate statistical test based on the normality of the data and the homogeneity
    of variances across groups, utilizing t-tests, Mann-Whitney U tests, ANOVA, or Kruskal-Wallis tests as appropriate.

    Parameters
    ----------
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

    Returns
    -------
    output_info : dict
        A dictionary containing the results of the hypothesis test, including test statistics, p-values,
        conclusions regarding the differences between groups, the name of the test used, and the assumptions
        tested (normality and equal variances).
    """

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


def predictor_core_categorical(contingency_table: pd.DataFrame, chi2_viability: bool, barnard_viability: bool, boschloo_viability: bool, fisher_viability: bool, yates_correction_viability: bool, alternative: str = 'two-sided'):
    """
    Conducts categorical hypothesis testing using contingency tables and appropriate statistical tests.

    This function assesses the association between two categorical variables by applying a series of statistical
    tests. It evaluates the data's suitability for different tests based on the shape of the contingency table,
    the minimum expected and observed frequencies, and specific methodological preferences, including Yates' correction
    for chi-square tests and alternatives for exact tests.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        A contingency table of the two categorical variables.
    chi2_viability : bool
        Indicates whether chi-square tests should be considered based on the data's suitability.
    barnard_viability, boschloo_viability, fisher_viability : bool
        Indicators for the applicability of Barnard's, Boschloo's, and Fisher's exact tests, respectively.
    yates_correction_viability : bool
        Determines whether Yates' correction is applicable based on the contingency table's shape and sample size.
    alternative : str, optional
        Specifies the alternative hypothesis for exact tests. Options include 'two-sided', 'less', or 'greater'.
        Defaults to 'two-sided'.

    Returns
    -------
    output_info : dict
        A comprehensive dictionary detailing the outcomes of the statistical tests performed, including test names,
        statistics, p-values, and conclusions about the association between the categorical variables. The dictionary
        also contains specific details about the application of Yates' correction and the chosen alternative hypothesis
        for exact tests.
    """

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
            test_name = f"Chi-square test (with Yates' Correction)"
            chi2_tip = f"\n\n☻ Tip: The Chi-square test of independence with Yates' Correction is used for 2x2 contingency tables with small sample sizes. Yates' Correction makes the test more conservative, reducing the Type I error rate by adjusting for the continuity of the chi-squared distribution. This correction is typically applied when sample sizes are small (often suggested for total sample sizes less than about 40), aiming to avoid overestimation of statistical significance."
            conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f})." if p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f})."
            chi2_output_info = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion, 'yates_correction': yates_correction_shape_bool, 'tip': chi2_tip, 'test_name': test_name}
            output_info['chi2_contingency'] = chi2_output_info
        else:
            stat, p_val, dof, expected_frequencies = chi2_contingency(contingency_table, correction=False)
            test_name = f"Chi-square test (without Yates' Correction)"
            chi2_tip = f"\n\n☻ Tip: The Chi-square test of independence without Yates' Correction is preferred when analyzing larger contingency tables or when sample sizes are sufficiently large, even for 2x2 tables (often suggested for total sample sizes greater than 40). Removing Yates' Correction can increase the test's power by not artificially adjusting for continuity, making it more sensitive to detect genuine associations between variables in settings where the assumptions of the chi-squared test are met."
            conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f})." if p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f})."
            chi2_output_info = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion, 'yates_correction': yates_correction_shape_bool, 'tip': chi2_tip, 'test_name': test_name}
            output_info['chi2_contingency'] = chi2_output_info
    else:
        if barnard_viability:
            barnard_test = barnard_exact(contingency_table, alternative=alternative.lower())
            barnard_stat = barnard_test.statistic
            barnard_p_val = barnard_test.pvalue
            bernard_test_name = f"Barnard's exact test ({alternative.lower()})"
            bernard_tip = f"\n\n☻ Tip: Barnard's exact test is often preferred for its power, especially in unbalanced designs or with small sample sizes, without the need for the continuity correction that Fisher's test applies."
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
            boschloo_tip = f"\n\n☻ Tip: Boschloo's exact test is an extension that increases power by combining the strengths of Fisher's and Barnard's tests, focusing on the most extreme probabilities."
            if alternative.lower() == 'two-sided':
                boschloo_conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {boschloo_p_val:.3f})." if boschloo_p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {boschloo_p_val:.3f})."
            elif alternative.lower() == 'less':
                boschloo_conclusion = f"The data do not support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})." if boschloo_p_val > 0.05 else f"The data support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})."
            elif alternative.lower() == 'greater':
                boschloo_conclusion = f"The data do not support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})." if boschloo_p_val > 0.05 else f"The data support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})."

            # consolidate info for ouput
            boschloo_output_info = {'stat': boschloo_stat, 'p_val': boschloo_p_val, 'conclusion': boschloo_conclusion, 'alternative': alternative.lower(), 'tip': boschloo_tip, 'test_name': boschloo_test_name}
            output_info['boschloo_exact'] = boschloo_output_info

        if fisher_viability:
            fisher_test = fisher_exact(contingency_table, alternative=alternative.lower())
            fisher_stat = fisher_test[0]
            fisher_p_val = fisher_test[1]
            fisher_test_name = f"Fisher's exact test ({alternative.lower()})"
            fisher_tip = f"\n☻ Tip: Fisher's exact test is traditionally used for small sample sizes, providing exact p-values under the null hypothesis of independence."
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
def predict_hypothesis(df: pd.DataFrame, var1: str, var2: str, normality_method: str = 'consensus', variance_method: str = 'consensus', exact_tests_alternative: str = 'two-sided', yates_min_sample_size: int = 40):
    """
    Automatically selects and performs the appropriate hypothesis test based on input variables from a DataFrame.
    This function simplifies hypothesis testing to requiring only two variables and a DataFrame, intelligently
    determining the test type, assessing necessary assumptions, and providing detailed test outcomes and conclusions.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data for hypothesis testing. This function analyzes the data to
        determine the type of hypothesis testing required based on the data types and relationships of `var1` and `var2`.
    var1 : str
        The name of the first variable for hypothesis testing. The combination of `var1` and `var2` data types
        determines the type of hypothesis test to perform. A pairing with two categorical variables
        triggers categorical testing (e.g., Chi-square, Fisher's exact test, etc.), while a numerical and
        categorical variable pairing, interpreted as target and grouping variables respectively, leads to numerical testing
        (e.g., t-tests, ANOVA, etc.). Numerical and numerical variable pairings are not supported.
    var2 : str
        The name of the second variable for hypothesis testing. Similar to `var1`, its combination with `var1`
        guides the selection of hypothesis testing procedures.
    normality_method : str, optional
        Specifies the method to evaluate normality within numerical hypothesis testing. Understanding the
        distribution is crucial to selecting parametric or non-parametric tests. Available methods include:
            - 'shapiro': Shapiro-Wilk test.
            - 'anderson': Anderson-Darling test.
            - 'normaltest': D’Agostino and Pearson’s test.
            - 'lilliefors': Lilliefors test for normality.
            - 'consensus': Utilizes a combination of the above tests to reach a consensus on normality.
        Defaults to 'consensus'. For detailed explanation, refer to `evaluate_normality`'s docstring.
    variance_method : str, optional
        Determines the method to evaluate variance homogeneity (equal variances) across groups in numerical hypothesis testing.
        The choice of test affects the selection between parametric and non-parametric tests for numerical data. Available methods include:
            - 'levene': Levene's test, robust to non-normal distributions.
            - 'bartlett': Bartlett’s test, sensitive to non-normal distributions.
            - 'fligner': Fligner-Killeen test, a non-parametric alternative.
            - 'consensus': A combination approach to determine equal variances across methods.
        Defaults to 'consensus'. For more information, see `evaluate_variance`'s docstring.
    exact_tests_alternative : str, optional
        For categorical hypothesis testing, this parameter specifies the alternative hypothesis direction for exact tests:
            - 'two-sided': Tests for any difference between the two variables without directionality.
            - 'less': Tests if the first variable is less than the second variable.
            - 'greater': Tests if the first variable is greater than the second variable.
        This parameter influences tests like Fisher's exact test or Barnard's exact test. Defaults to 'two-sided'.
    yates_min_sample_size : int, optional
        Specifies the minimum sample size threshold for applying Yates' correction in chi-square testing to adjust
        for continuity. The correction is applied to 2x2 contingency tables with small sample sizes to prevent
        overestimation of the significance level. Defaults to 40.

    Returns
    -------
    output_info : dict
        A dictionary with key the short test name (e.g. f_oneway) and value another dictionary which contains all results from that test, namely:
            - 'stat': The test statistic value, quantifying the degree to which the observed data conform to the null hypothesis.
            - 'p_val': The p-value, indicating the probability of observing the test results under the null hypothesis.
            - 'conclusion': A textual interpretation of the test outcome, stating whether the evidence was sufficient to reject the null hypothesis.
            - 'test_name': The full name of the statistical test performed (e.g., 'Independent Samples T-Test', 'Chi-square test').
        Additional keys in certain scenarios may be:
            - 'alternative': Specifies the alternative hypothesis direction used in exact tests ('two-sided', 'less', 'greater').
            - 'yates_correction': A boolean that indicates whether a Yate's correction was applied used in Chi-square test.
            - 'normality': A boolean that indicates whether the data were found to meet the normality assumption.
            - 'equal_variance': A boolean that indicates whether the data were found to have equal variances across groups.
            - 'tip': Helpful insights or considerations regarding the test's application or interpretation.

    Notes
    -----
    `predict_hypothesis` is engineered to facilitate an intuitive yet powerful entry into hypothesis testing. Here’s a deeper look into its operational logic:

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
        The function demystifies statistical analysis, making it approachable for users across various disciplines, enabling informed decisions based on robust statistical evidence.

    This function stands out by automating complex decision trees involved in statistical testing, offering a
    simplified yet comprehensive approach to hypothesis testing. It exemplifies how advanced statistical analysis
    can be made accessible and actionable, fostering data-driven decision-making.

    Examples
    --------
    # First, we'll create a DataFrame with categorical and numerical variables to use in our examples:
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
            'Group': np.random.choice(['Control', 'Treatment'], size=100),
            'Score': np.random.normal(0, 1, 100),
            'Category': np.random.choice(['Type1', 'Type2'], size=100),
            'Feature2': np.random.exponential(1, 100)
        })

    # Scenario 1: Basic numerical hypothesis testing (T-test or ANOVA based on groups)
    >>> output_num_basic = predict_hypothesis(df, 'Group', 'Score')

    # Scenario 2: Numerical hypothesis testing specifying method to evaluate normality
    >>> output_num_normality = predict_hypothesis(df, 'Group', 'Score', normality_method='shapiro')

    # Scenario 3: Numerical hypothesis testing with a specified method to evaluate variance
    >>> output_num_variance = predict_hypothesis(df, 'Group', 'Score', variance_method='levene')

    # Scenario 4: Categorical hypothesis testing (Chi-square or Fisher's exact test)
    >>> output_cat_basic = predict_hypothesis(df, 'Group', 'Category')

    # Scenario 5: Categorical hypothesis testing with alternative hypothesis specified
    >>> output_cat_alternative = predict_hypothesis(df, 'Category', 'Group', exact_tests_alternative='less')

    # Scenario 6: Applying Yates' correction in a Chi-square test for small samples
    >>> output_yates_correction = predict_hypothesis(df, 'Group', 'Category', yates_min_sample_size=30)

    # Scenario 7: Comprehensive numerical hypothesis testing using consensus for normality and variance evaluation
    >>> output_num_comprehensive = predict_hypothesis(df, 'Group', 'Score', normality_method='consensus', variance_method='consensus')

    # Scenario 8: Testing with a numerical variable against a different grouping variable
    >>> output_different_group = predict_hypothesis(df, 'Feature2', 'Group')

    # Scenario 9: Exploring exact tests in categorical hypothesis testing for a 2x2 table
    >>> df_small = df.sample(20) # Smaller sample for demonstration
    >>> output_exact_tests = predict_hypothesis(df_small, 'Category', 'Group', exact_tests_alternative='two-sided')
    """

    # determine appropriate testing procedure and variable interpretation
    data_types = evaluate_dtype(df, [var1, var2])

    if data_types[var1] == 'categorical' and data_types[var2] == 'numerical':
        hypothesis_testing = 'numerical'
        grouping_variable = var1
        target_variable = var2
        print(f"< INITIALIZING predict_hypothesis() >\n")
        print(f"Performing {hypothesis_testing} hypothesis testing with:")
        print(f"  ➡ Grouping variable: '{grouping_variable}' (with groups: {df[grouping_variable].unique()})\n  ➡ Target variable: '{target_variable}'\n")
        print(f"Output Contents:\n (1) Results of Normality Testing\n (2) Results of Variance Testing\n (3) Results of Hypothesis Testing\n\n")
    elif data_types[var1] == 'numerical' and data_types[var2] == 'categorical':
        hypothesis_testing = 'numerical'
        grouping_variable = var2
        target_variable = var1
        print(f"< INITIALIZING predict_hypothesis() >\n")
        print(f"Performing {hypothesis_testing} hypothesis testing with:")
        print(f"  ➡ Grouping variable: '{grouping_variable}' (with groups: {df[grouping_variable].unique()})\n  ➡ Target variable: '{target_variable}'\n")
        print(f"Output Contents:\n (1) Results of Normality Testing\n (2) Results of Variance Testing\n (3) Results of Hypothesis Testing\n\n")
    elif data_types[var1] == 'categorical' and data_types[var2] == 'categorical':
        hypothesis_testing = 'categorical'
        categorical_variable1 = var1
        categorical_variable2 = var2
        print(f"< INITIALIZING predict_hypothesis() >")
        print(f"Performing {hypothesis_testing} hypothesis testing with:\n")
        print(f"  ➡ Categorical variable 1: '{categorical_variable1}'\n  ➡ Categorical variable 2: '{categorical_variable2}'\n\n")

    # perform appropriate hypothesis testing process
    if hypothesis_testing == 'numerical':
        # evaluate test assumptions
        normality_bool = evaluate_normality(df, target_variable, grouping_variable, method=normality_method, pipeline=True)
        equal_variance_bool = evaluate_variance(df, target_variable, grouping_variable, normality_info=normality_bool, method=variance_method, pipeline=True)

        # perform hypothesis testing
        output_info = predictor_core_numerical(df, target_variable, grouping_variable, normality_bool, equal_variance_bool)
        return output_info
    elif hypothesis_testing == 'categorical':
        # create contingency table
        contingency_table = pd.crosstab(df[categorical_variable1], df[categorical_variable2])

        # evaluate test criteria
        chi2_viability, yates_correction_viability, barnard_viability, boschloo_viability, fisher_viability = evaluate_contingency_table(contingency_table, min_sample_size_yates=yates_min_sample_size, pipeline=True, quiet=True)

        # perform hypothesis testing
        output_info = predictor_core_categorical(contingency_table, chi2_viability, barnard_viability, boschloo_viability, fisher_viability, yates_correction_viability, alternative=exact_tests_alternative)
        return output_info


# smoke tests


# create df for testing
data = {
    'Category1': np.random.choice(['Apple', 'Banana', 'Cherry'], size=100),
    'Category2': np.random.choice(['Yes', 'No'], size=100),
    'Category3': np.random.choice(['Low', 'Medium', 'High'], size=100),
    'Category4': np.random.choice(['Short', 'Tall'], size=100),
    'Feature1': np.random.normal(0, 1, 100),
    'Feature2': np.random.exponential(1, 100),
    'Feature3': np.random.randint(1, 100, 100)
}

test_df = pd.DataFrame(data)

output_info = predict_hypothesis(test_df, 'Category1', 'Category2')
