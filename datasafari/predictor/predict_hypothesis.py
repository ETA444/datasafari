import numpy as np
import pandas as pd
from scipy.stats import (
    ttest_ind, mannwhitneyu, f_oneway, kruskal,  # numerical hypothesis testing
    chi2_contingency, barnard_exact, boschloo_exact, fisher_exact  # categorical hypothesis testing
)
from scipy.stats.contingency import expected_freq
from datasafari.evaluator import evaluate_normality, evaluate_variance


# Utility functions: used only in this module
def evaluate_data_types(df, cols):
    """Evaluates the data types of provided columns to choose the appropriate hypothesis testing procedure."""

    data_type_dictionary = {}
    for col in cols:
        if df[col].dtype in ['int', 'float']:
            data_type_dictionary[col] = 'numeric'
        else:
            data_type_dictionary[col] = 'categorical'
    return data_type_dictionary


def evaluate_frequencies(df: pd.DataFrame, categorical_variable1: str, categorical_variable2: str):
    """Evaluates whether expected and observed frequencies are adequate for chi2_contingency()."""

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


def evaluate_shape(contingency_table):
    """Evaluates whether the shape of the contingency table is suitable and for what tests."""

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


def predict_hypo_num(df, target_variable, grouping_variable, normality_bool, equal_variances_bool):
    """Runs tests for a hypothesis for numerical data."""

    groups = df[grouping_variable].unique().tolist()
    samples = [df[df[grouping_variable] == group][target_variable] for group in groups]

    # use evaluators to get normality and equal variance - THIS CODE IS FOR predict_hypothesis() SO THAT WE CAN HAVE custom normality_method, etc..
    #  normality_bool = evaluate_normality(df, target_variable, grouping_variable, method='consensus', pipeline=True)
    #  equal_variances_bool = evaluate_variance(df, target_variable, grouping_variable, normality_info=normality_bool, method='consensus', pipeline=True)

    # define output object
    output_info = {}

    if len(samples) == 2:  # two sample testing
        if normality_bool and equal_variances_bool:  # parametric testing
            stat, p_val = ttest_ind(*samples)
            test_name = 'Independent Samples T-Test'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference between the group means, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference between the group means, rejecting the null hypothesis."
            output_info['ttest_ind'] = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion}
        else:  # non-parametric testing
            stat, p_val = mannwhitneyu(*samples)
            test_name = 'Mann-Whitney U Rank Test (Two Independent Samples)'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference in group distributions, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference in group distributions, rejecting the null hypothesis."
            output_info['mannwhitneyu'] = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion}
    else:  # more than two samples
        if normality_bool and equal_variances_bool:
            stat, p_val = f_oneway(*samples)
            test_name = f'One-way ANOVA (with {len(samples)} groups)'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference among the group means, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference among the group means, rejecting the null hypothesis."
            output_info['f_oneway'] = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion}
        else:
            stat, p_val = kruskal(*samples)
            test_name = f'Kruskal-Wallis H-test (with {len(samples)} groups)'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference among the group distributions, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference among the group distributions, rejecting the null hypothesis."
            output_info['kruskal'] = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion}

    # construct console output and return
    print(f"< HYPOTHESIS TESTING: {test_name}>\nBased on:\n  ➡ Normality assumption: {'✔' if normality_bool else '✘'}\n  ➡ Equal variances assumption: {'✔' if equal_variances_bool else '✘'}\n  ➡ Nr. of Groups: {len(samples)} groups\n  ∴ predict_hypothesis() is performing {test_name}:\n")
    print(f"Results of {test_name}:\n  ➡ statistic: {stat}\n  ➡ p-value: {p_val}\n  ∴ Conclusion: {conclusion}\n")
    return output_info


def predict_hypo_cat(contingency_table, chi2_bool, barnard_bool, boschloo_bool, fisher_bool, yates_correction_shape_bool, alternative: str = 'two-sided', yates_min_sample_size: int = 40):
    """Runs tests for a hypothesis for categorical data."""

    # to avoid unnecessary parameter inputs use contingency_table object
    categorical_variable1 = contingency_table.index.name
    categorical_variable2 = contingency_table.columns.name
    min_expected_frequency = expected_freq(contingency_table).min()
    min_observed_frequency = contingency_table.min().min()
    contingency_table_shape = contingency_table.shape
    sample_size = np.sum(contingency_table.values)

    # define output object
    output_info = {}

    if chi2_bool:
        if yates_correction_shape_bool and sample_size <= yates_min_sample_size:
            stat, p_val, dof, expected_frequencies = chi2_contingency(contingency_table, correction=True)
            test_name = f"Chi-square test (with Yates' Correction)"
            chi2_tip = f"\n☻ Tip: The Chi-square test of independence with Yates' Correction is used for 2x2 contingency tables with small sample sizes. Yates' Correction makes the test more conservative, reducing the Type I error rate by adjusting for the continuity of the chi-squared distribution. This correction is typically applied when sample sizes are small (often suggested for total sample sizes less than about 40), aiming to avoid overestimation of statistical significance."
            conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f}, with Yate's Correction)." if p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f}, with Yate's Correction)."
            chi2_output_info = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion, 'tip': chi2_tip, 'test_name': test_name}
            output_info['chi2_contingency'] = chi2_output_info
        else:
            stat, p_val, dof, expected_frequencies = chi2_contingency(contingency_table, correction=False)
            test_name = f"Chi-square test (without Yates' Correction)"
            chi2_tip = f"\n☻ Tip: The Chi-square test of independence without Yates' Correction is preferred when analyzing larger contingency tables or when sample sizes are sufficiently large, even for 2x2 tables (often suggested for total sample sizes greater than 40). Removing Yates' Correction can increase the test's power by not artificially adjusting for continuity, making it more sensitive to detect genuine associations between variables in settings where the assumptions of the chi-squared test are met."
            conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f}, without Yate's Correction)." if p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f}, without Yate's Correction)."
            chi2_output_info = {'stat': stat, 'p_val': p_val, 'conclusion': conclusion, 'tip': chi2_tip, 'test_name': test_name}
            output_info['chi2_contingency'] = chi2_output_info
    else:
        if barnard_bool:
            barnard_test = barnard_exact(contingency_table, alternative=alternative.lower())
            barnard_stat = barnard_test.statistic
            barnard_p_val = barnard_test.pvalue
            bernard_test_name = f"Barnard's exact test ({alternative.lower()})"
            bernard_tip = f"\n☻ Tip: Barnard's exact test is often preferred for its power, especially in unbalanced designs or with small sample sizes, without the need for the continuity correction that Fisher's test applies."
            if alternative.lower() == 'two-sided':
                bernard_conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {barnard_p_val:.3f})." if barnard_p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {barnard_p_val:.3f})."
            elif alternative.lower() == 'less':
                bernard_conclusion = f"The data do not support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {barnard_p_val:.3f})." if barnard_p_val > 0.05 else f"The data support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {barnard_p_val:.3f})."
            elif alternative.lower() == 'greater':
                bernard_conclusion = f"The data do not support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {barnard_p_val:.3f})." if barnard_p_val > 0.05 else f"The data support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {barnard_p_val:.3f})."

            # consolidate info for output
            bernard_output_info = {'stat': barnard_stat, 'p_val': barnard_p_val, 'conclusion': bernard_conclusion, 'tip': bernard_tip, 'test_name': bernard_test_name}
            output_info['barnard_exact'] = bernard_output_info

        if boschloo_bool:
            boschloo_test = boschloo_exact(contingency_table, alternative=alternative.lower())
            boschloo_stat = boschloo_test.statistic
            boschloo_p_val = boschloo_test.pvalue
            boschloo_test_name = f"Boschloo's exact test ({alternative.lower()})"
            boschloo_tip = f"\n☻ Tip: Boschloo's exact test is an extension that increases power by combining the strengths of Fisher's and Barnard's tests, focusing on the most extreme probabilities."
            if alternative.lower() == 'two-sided':
                boschloo_conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {boschloo_p_val:.3f})." if boschloo_p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {boschloo_p_val:.3f})."
            elif alternative.lower() == 'less':
                boschloo_conclusion = f"The data do not support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})." if boschloo_p_val > 0.05 else f"The data support a statistically significant decrease in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})."
            elif alternative.lower() == 'greater':
                boschloo_conclusion = f"The data do not support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})." if boschloo_p_val > 0.05 else f"The data support a statistically significant increase in the frequency of {categorical_variable1} compared to {categorical_variable2} (p = {boschloo_p_val:.3f})."

            # consolidate info for ouput
            boschloo_output_info = {'stat': boschloo_stat, 'p_val': boschloo_p_val, 'conclusion': boschloo_conclusion, 'tip': boschloo_tip, 'test_name': boschloo_test_name}
            output_info['boschloo_exact'] = boschloo_output_info

        if fisher_bool:
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
            fisher_output_info = {'stat': fisher_stat, 'p_val': fisher_p_val, 'conclusion': fisher_conclusion, 'tip': fisher_tip, 'test_name': fisher_test_name}
            output_info['fisher_exact'] = fisher_output_info

    # console output & return
    generate_test_names = (output_info[0]['test_name'] if len(output_info) == 1 else [info['test_name'] for info in output_info.values()])
    print(f"< HYPOTHESIS TESTING: {generate_test_names}>\nBased on:\n  ➡ Sample size: {sample_size}\n  ➡ Minimum Observed Frequency: {min_observed_frequency}\n  ➡ Minimum Expected Frequency: {min_expected_frequency}\n  ➡ Contingency table shape: {contingency_table_shape[0]}x{contingency_table_shape[1]}\n  ∴ Performing {generate_test_names}:\n")
    print("\n☻ Tip: Consider the tips provided for each test and assess which of the exact tests provided is most suitable for your data.") if len(output_info) > 1 else print('')
    [print(f"Results of {info['test_name']}:\n  ➡ statistic: {info['stat']}\n  ➡ p-value: {info['p_val']}\n  ∴ Conclusion: {info['conclusion']}{info['tip']}\n") for info in output_info.values()]
    return output_info


# Main function
def predict_hypothesis():
    """
    """

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
contingency_table_test2x2 = pd.crosstab(test_df['Category2'], test_df['Category4'])
contingency_table_test3x3 = pd.crosstab(test_df['Category1'], test_df['Category3'])
contingency_table_test2x2sample39 = pd.crosstab(test_df['Category2'][0:39], test_df['Category4'][0:39])
grouping = 'Category2'
variable = 'Feature1'

# TODO: Construct main function
# TODO: Write Numpy Docstring for main function
