import numpy as np
import pandas as pd
from scipy.stats import (
    shapiro, anderson, normaltest,  # normality testing
    levene, bartlett, fligner,  # equal variances testing
    ttest_ind, mannwhitneyu, f_oneway, kruskal,  # numerical hypothesis testing
    chi2_contingency, barnard_exact, boschloo_exact, fisher_exact  # categorical hypothesis testing
)
from scipy.stats.contingency import expected_freq
from statsmodels.stats.diagnostic import lilliefors


def predict_hypothesis():
    """
    The general idea here is to create a function where:
    (1) Adheres to MVP of simplicity in user-input
    (2) In case of hypothesis it does everything automatically (e.g. for numeric data it will check
    for normality and choose parametric or non-parametric tests etc. it will also reject or not the null itself)
    (3) It therefore not only eases writing the code but also decision-making, the user does not need
    to know about: what test to use based on the data, what test of the appropriate tests are better (all are provided),
    and so on
    (4) It handles the key datatypes categorical and numeric automatically

    Schema of the idea:
    > In > user data (df and ??)
    | 1 Inside: infer numeric or categorical
    \\
     \\
        | 2A Inside: if numeric test assumptions to choose if parametric or non-parametric
        | 3A Inside: perform either set of tests and get results
        | 4A Inside: use results to build output and make a ready conclusion for the user
        < Out < results + conclusion
    \\
     \\
        | 2B Inside: if categorical  ...
        | 3B Inside: ...
        | 4B Inside: ...
        < Out < results + conclusion
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


# infer data type
def assign_data_types(df, cols):
    data_type_dictionary = {}
    for col in cols:
        if df[col].dtype in ['int', 'float']:
            data_type_dictionary[col] = 'numeric'
        else:
            data_type_dictionary[col] = 'categorical'
    return data_type_dictionary


dt_dict = assign_data_types(test_df, test_df.columns)


def evaluate_normality(df, target_variable, grouping_variable, method: str = 'consensus', pipeline: bool = False):
    groups = df[grouping_variable].unique().tolist()

    # define output objects
    output_info = {}  # non-pipeline
    normality_info = {}  # pipeline single-method
    output_consensus = None  # pipeline consensus method

    if method in ['shapiro', 'consensus']:
        # calculating statistic and p-values to define normality
        shapiro_stats = [shapiro(df[df[grouping_variable] == group][target_variable]).statistic for group in groups]
        shapiro_pvals = [shapiro(df[df[grouping_variable] == group][target_variable]).pvalue for group in groups]
        shapiro_normality = [True if p > 0.05 else False for p in shapiro_pvals]

        # save the info for return and text for output
        shapiro_info = {group: {'stat': shapiro_stats[n], 'p': shapiro_pvals[n], 'normality': shapiro_normality[n]} for n, group in enumerate(groups)}
        shapiro_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in shapiro_info.items()]
        shapiro_title = f"< NORMALITY TESTING: SHAPIRO-WILK >\n\n"
        shapiro_tip = "☻ Tip: The Shapiro-Wilk test is particularly well-suited for small sample sizes (n < 50) and is considered one of the most powerful tests for assessing normality. It is sensitive to departures from normality, making it a preferred choice for rigorous normality assessments in small datasets.\n"

        # save info in return object and conditionally print to console
        output_info['shapiro'] = shapiro_info
        normality_info['shapiro_group_consensus'] = all(shapiro_normality)

        # end it here if non-consensus method
        if method == 'shapiro':
            print(shapiro_title, *shapiro_text, shapiro_tip)
            return output_info if not pipeline else normality_info['shapiro_group_consensus']

    if method in ['anderson', 'consensus']:
        # calculating statistic to define normality
        anderson_stats = [anderson(df[df[grouping_variable] == group][target_variable]).statistic for group in groups]
        anderson_critical_values = [anderson(df[df[grouping_variable] == group][target_variable]).critical_values[2] for group in groups]
        anderson_normality = [True if c_val > anderson_stats[n] else False for n, c_val in enumerate(anderson_critical_values)]

        # save the info for return and text for output
        anderson_info = {group: {'stat': anderson_stats[n], 'p': anderson_critical_values[n], 'normality': anderson_normality[n]} for n, group in enumerate(groups)}
        anderson_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in anderson_info.items()]
        anderson_title = f"< NORMALITY TESTING: ANDERSON-DARLING >\n\n"
        anderson_tip = "☻ Tip: The Anderson-Darling test is a versatile test that can be applied to any sample size and is especially useful for comparing against multiple distribution types, not just the normal. It places more emphasis on the tails of the distribution than the Shapiro-Wilk test, making it useful for detecting outliers or heavy-tailed distributions.\n"

        # saving info
        output_info['anderson'] = anderson_info
        normality_info['anderson_group_consensus'] = all(anderson_normality)

        # end it here if non-consensus method
        if method == 'anderson':
            print(anderson_title, *anderson_text, anderson_tip)
            return output_info if not pipeline else normality_info['anderson_group_consensus']

    if method in ['normaltest', 'consensus']:
        # calculating statistic and p-values to define normality
        normaltest_stats = [normaltest(df[df[grouping_variable] == group][target_variable]).statistic for group in groups]
        normaltest_pvals = [normaltest(df[df[grouping_variable] == group][target_variable]).pvalue for group in groups]
        normaltest_normality = [True if p > 0.05 else False for p in normaltest_pvals]

        # save the info for return and text for output
        normaltest_info = {group: {'stat': normaltest_stats[n], 'p': normaltest_pvals[n], 'normality': normaltest_normality[n]} for n, group in enumerate(groups)}
        normaltest_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in normaltest_info.items()]
        normaltest_title = f"< NORMALITY TESTING: D'AGOSTINO-PEARSON NORMALTEST >\n\n"
        normaltest_tip = "☻ Tip: The D'Agostino-Pearson normality test, or simply 'normaltest', is best applied when the sample size is larger, as it combines skewness and kurtosis to form a test statistic. This test is useful for detecting departures from normality that involve asymmetry and tail thickness, offering a good balance between sensitivity and specificity in medium to large sample sizes.\n"

        # saving info
        output_info['normaltest'] = normaltest_info
        normality_info['normaltest_group_consensus'] = all(normaltest_normality)

        # end it here if non-consensus method
        if method == 'normaltest':
            print(normaltest_title, *normaltest_text, normaltest_tip)
            return output_info if not pipeline else normality_info['normaltest_group_consensus']

    if method in ['lilliefors', 'consensus']:
        # calculating statistic and p-values to define normality
        lilliefors_stats = [lilliefors(df[df[grouping_variable] == group][target_variable])[0] for group in groups]
        lilliefors_pvals = [lilliefors(df[df[grouping_variable] == group][target_variable])[1] for group in groups]
        lilliefors_normality = [True if p > 0.05 else False for p in lilliefors_pvals]

        # save the info for return and text for output
        lilliefors_info = {group: {'stat': lilliefors_stats[n], 'p': lilliefors_pvals[n], 'normality': lilliefors_normality[n]} for n, group in enumerate(groups)}
        lilliefors_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in lilliefors_info.items()]
        lilliefors_title = f"< NORMALITY TESTING: LILLIEFORS' TEST >\n\n"
        lilliefors_tip = "☻ Tip: The Lilliefors test is an adaptation of the Kolmogorov-Smirnov test for normality with the benefit of not requiring the mean and variance to be known parameters. It's particularly useful for small to moderately sized samples and is sensitive to deviations from normality in the center of the distribution rather than the tails. This makes it complementary to tests like the Anderson-Darling when a comprehensive assessment of normality is needed.\n"

        # saving info
        output_info['lilliefors'] = lilliefors_info
        normality_info['lilliefors_group_consensus'] = all(lilliefors_normality)

        # end here if non-consensus method
        if method == 'lilliefors':
            print(lilliefors_title, *lilliefors_text, lilliefors_tip)
            return output_info if not pipeline else normality_info['lilliefors_group_consensus']

    if method == 'consensus':
        # the logic is that more than half of the tests need to give True for the consensus to be True
        true_count = [group_normality_consensus for test_name, group_normality_consensus in normality_info.items()].count(True)
        false_count = [group_normality_consensus for test_name, group_normality_consensus in normality_info.items()].count(False)
        half_point = len(normality_info)/2
        if true_count > half_point:
            consensus_percent = (true_count / len(normality_info)) * 100
            normality_consensus_result = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest Normality. *\n\n* Detailed results of each test are provided below.\n"

            # used only within predict_hypothesis() pipeline
            output_consensus = True

        elif true_count < half_point:
            consensus_percent = (false_count / len(normality_info)) * 100
            normality_consensus_result = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest Non-Normality. *\n\n* Detailed results of each test are provided below:\n"

            # used only within predict_hypothesis() pipeline
            output_consensus = False

        elif true_count == half_point:
            # the logic is that we default to the decisions of shapiro-wilk and anderson-darling (if both true then true) - internal use only
            normality_consensus_result = f"  ➡ Result: Consensus is not reached. (50% Normality / 50% Non-normality)\n\n∴ Please refer to the results of each test below:\n"

            # used only within predict_hypothesis() pipeline
            output_consensus = all([normality_info['shapiro_group_consensus'], normality_info['anderson_group_consensus']])

        # construct output and return
        print(f"< NORMALITY TESTING: CONSENSUS >\nThe consensus method bases its conclusion on 4 tests: Shapiro-Wilk test, Anderson-Darling test, D'Agostino-Pearson test, Lilliefors test. (Note: More than 50% must have the same outcome to reach consensus. In predict_hypothesis(), if consensus is not reached normality is settled based on Shapiro-Wilk and Anderson-Darling outcomes)\n\n{normality_consensus_result}")
        print(shapiro_title, *shapiro_text, shapiro_tip)
        print(anderson_title, *anderson_text, anderson_tip)
        print(normaltest_title, *normaltest_text, normaltest_tip)
        print(lilliefors_title, *lilliefors_text, lilliefors_tip)
        return output_info if not pipeline else output_consensus


evaluate_normality(test_df, variable, grouping, method='lilliefors', pipeline=True)


def evaluate_variance(df, target_variable, grouping_variable, normality_info: bool = None, method: str = 'consensus', pipeline: bool = False):
    groups = df[grouping_variable].unique().tolist()
    samples = [df[df[grouping_variable] == group][target_variable] for group in groups]

    # define output objects
    output_info = {}  # non-pipeline
    variance_info = {}  # pipeline single-method
    output_consensus = None  # pipeline consensus method

    if method in ['levene', 'consensus']:
        levene_stat, levene_pval = levene(*samples).statistic, levene(*samples).pvalue
        variance_info['levene'] = levene_pval > 0.05

        # save the info in return object
        output_info['levene'] = {'stat': levene_stat, 'p': levene_pval, 'equal_variances': variance_info['levene']}

        # construct console output
        levene_text = f"Results for samples in groups of '{grouping_variable}' for ['{target_variable}'] target variable:\n  ➡ statistic: {levene_stat}\n  ➡ p-value: {levene_pval}\n{(f'  ∴ Equal variances: Yes (H0 cannot be rejected)' if variance_info['levene'] else f'  ∴ Equal variances: No (H0 rejected)')}\n\n"
        levene_title = f"< EQUAL VARIANCES TESTING: LEVENE >\n\n"
        levene_tip = "☻ Tip: The Levene test is robust against departures from normality, making it suitable for a wide range of distributions, including those that are not normally distributed. It is preferred when you suspect that your data may not adhere to the normality assumption.\n"

        # output & return
        if method == 'levene':
            print(levene_title, levene_text, levene_tip)
            return output_info if not pipeline else variance_info['levene']

    if method in ['fligner', 'consensus']:
        fligner_stat, fligner_pval = fligner(*samples).statistic, fligner(*samples).pvalue
        variance_info['fligner'] = fligner_pval > 0.05

        # save the info in return object
        output_info['fligner'] = {'stat': fligner_stat, 'p': fligner_pval, 'equal_variances': variance_info['fligner']}

        # construct console output
        fligner_text = f"Results for samples in groups of '{grouping_variable}' for ['{target_variable}'] target variable:\n  ➡ statistic: {fligner_stat}\n  ➡ p-value: {fligner_pval}\n{(f'  ∴ Equal variances: Yes (H0 cannot be rejected)' if variance_info['fligner'] else f'  ∴ Equal variances: No (H0 rejected)')}\n\n"
        fligner_title = f"< EQUAL VARIANCES TESTING: FLIGNER-KILLEEN >\n\n"
        fligner_tip = "☻ Tip: The Fligner-Killeen test is a non-parametric alternative that is less sensitive to departures from normality compared to the Bartlett test. It's a good choice when dealing with data that might not be normally distributed, offering robustness similar to the Levene test but through a different statistical approach.\n"

        # output & return
        if method == 'fligner':
            print(fligner_title, fligner_text, fligner_tip)
            return output_info if not pipeline else variance_info['fligner']

    if method in ['bartlett', 'consensus'] and normality_info:
        bartlett_stat, bartlett_pval = bartlett(*samples).statistic, bartlett(*samples).pvalue
        variance_info['bartlett'] = bartlett_pval > 0.05

        # save the info in return object
        output_info['bartlett'] = {'stat': bartlett_stat, 'p': bartlett_pval, 'equal_variances': variance_info['bartlett']}

        # construct console output
        bartlett_text = f"Results for samples in groups of '{grouping_variable}' for ['{target_variable}'] target variable:\n  ➡ statistic: {bartlett_stat}\n  ➡ p-value: {bartlett_pval}\n{(f'  ∴ Equal variances: Yes (H0 cannot be rejected)' if variance_info['bartlett'] else f'  ∴ Equal variances: No (H0 rejected)')}\n\n"
        bartlett_title = f"< EQUAL VARIANCES TESTING: BARTLETT >\n\n"
        bartlett_tip = "☻ Tip: The Fligner-Killeen test is a non-parametric alternative that is less sensitive to departures from normality compared to the Bartlett test. It's a good choice when dealing with data that might not be normally distributed, offering robustness similar to the Levene test but through a different statistical approach.\n"

        # output & return
        if method == 'bartlett':
            print(bartlett_title, bartlett_text, bartlett_tip)
            return output_info if not pipeline else variance_info['bartlett']

    if method == 'consensus':
        # the logic is that more than half of the tests need to give True for the consensus to be True
        variance_results = [variance_bool for variance_bool in variance_info.values()]
        true_count = variance_results.count(True)
        false_count = variance_results.count(False)
        half_point = 1.5 if len(variance_results) == 3 else 1
        if true_count > half_point:
            consensus_percent = (true_count / len(variance_results)) * 100
            variance_consensus_text = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest equal variance between samples. *\n\n* Detailed results of each test are provided below.\n"

            # used only within predict_hypothesis() pipeline
            variance_consensus = True

        elif true_count < half_point:
            consensus_percent = (false_count / len(variance_results)) * 100
            variance_consensus_text = f"  ➡ Result: Consensus is reached.\n  ➡ {consensus_percent}% of tests suggest unequal variance between samples. *\n\n* Detailed results of each test are provided below:\n"

            # used only within predict_hypothesis() pipeline
            variance_consensus = False

        elif true_count == half_point:
            # the logic is that we default to the decisions of levene - internal use only
            variance_consensus_text = f"  ➡ Result: Consensus is not reached.\n\n∴ Please refer to the results of each test below:\n"

            # used only within predict_hypothesis() pipeline
            variance_consensus = variance_info['levene']

        print(f"< VARIANCE TESTING: CONSENSUS >\nThe consensus method bases its conclusion on 2-3 tests: Levene test, Fligner-Killeen test, Bartlett test. (Note: More than 50% must have the same outcome to reach consensus.)\n\n{variance_consensus_text}")
        print(levene_title, levene_text, levene_tip)
        print(fligner_title, fligner_text, fligner_tip)
        print(bartlett_title, bartlett_text, bartlett_tip) if normality_info else f"\n\n< NOTE ON BARTLETT >\nBartlett was not used in consensus as no normality info has been provided or data is non-normal. Accuracy of Bartlett's test results rely heavily on normality."

        return output_info if not pipeline else variance_consensus


equal_variances_info = evaluate_variance(test_df, variable, grouping)

# DONE! TODO: simplify output_info structure on below func (considering only one test will be outputted
# TODO: make evaluate_shape stand on it's own two feet OR integrate into the performer
# TODO: make evaluate_frequency stand on it's own two feet OR integrate into the performer


def perform_hypothesis_testing_numeric(df, target_variable, grouping_variable, normality_bool, equal_variances_bool):
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


perform_hypothesis_testing_numeric(test_df, variable, grouping, normality_info, equal_variances_info)


def testing_frequencies(df, categorical_variable1, categorical_variable2):
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


def testing_shape(contingency_table):
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


def perform_hypothesis_testing_categorical(contingency_table, chi2_bool, barnard_bool, boschloo_bool, fisher_bool, yates_correction_shape_bool, alternative: str = 'two-sided', yates_min_sample_size: int = 40):
    # to avoid unnecessary parameter inputs get var names from contingency_table object
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


# testing exact tests
perform_hypothesis_testing_categorical(contingency_table_test2x2, False, True, True, True, True)

# testing chi2 with yates
perform_hypothesis_testing_categorical(contingency_table_test2x2sample39, True, False, False, False, True)

# testing chi2 without yates
perform_hypothesis_testing_categorical(contingency_table_test3x3, True, False, False, False, False)


    # DONE! TODO: Handle returns of numerical
    # TODO: Error handling for all
    # TOOD: Enrich numerical testing
    # TODO: Enrich parameters of internal functions
    # TODO: Construct main function
    # TODO: Write Numpy Docstring for main function
