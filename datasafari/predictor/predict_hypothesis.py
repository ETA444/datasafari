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


def test_normality(df, target_variable, grouping_variable, method: str = 'consensus', pipeline: bool = False):
    groups = df[grouping_variable].unique().tolist()

    # define output object
    output_info = {}
    normality_info = {}

    if method in ['shapiro_wilk', 'consensus']:
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
        output_info['shapiro_wilk'] = shapiro_info
        normality_info['shapiro_wilk_group_consensus'] = all(shapiro_normality)

        # end it here if non-consensus method
        if method == 'shapiro_wilk':
            print(shapiro_title, *shapiro_text, shapiro_tip)
            return output_info if not pipeline else normality_info['shapiro_wilk_group_consensus']

    if method in ['anderson_darling', 'consensus']:
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
        output_info['anderson_darling'] = anderson_info
        normality_info['anderson_darling_group_consensus'] = all(anderson_normality)

        # end it here if non-consensus method
        if method == 'anderson_darling':
            print(anderson_title, *anderson_text, anderson_tip)
            return output_info if not pipeline else normality_info['anderson_darling_group_consensus']

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
            output_consensus = all([normality_info['shapiro_wilk_group_consensus'], normality_info['anderson_darling_group_consensus']])

        # construct output and return
        print(f"< NORMALITY TESTING: CONSENSUS >\nThe consensus method bases its conclusion on 4 tests: Shapiro-Wilk test, Anderson-Darling test, D'Agostino-Pearson test, Lilliefors test. (Note: More than 50% must have the same outcome to reach consensus. In predict_hypothesis(), if consensus is not reached normality is settled based on Shapiro-Wilk and Anderson-Darling outcomes)\n\n{normality_consensus_result}")
        print(shapiro_title, *shapiro_text, shapiro_tip)
        print(anderson_title, *anderson_text, anderson_tip)
        print(normaltest_title, *normaltest_text, normaltest_tip)
        print(lilliefors_title, *lilliefors_text, lilliefors_tip)
        return output_info if not pipeline else output_consensus


test_normality(test_df, variable, grouping, method='lilliefors', pipeline=True)


# TODO: Implement test preference mechanism where users can choose more tests (e.g. bartlett)
def testing_equal_variances(df, target_variable, grouping_variable, method: str = 'shapiro'):
    groups = df[grouping_variable].unique().tolist()
    samples = [df[df[grouping_variable] == group][target_variable] for group in groups]

    # levene test #
    # calculating statistic and p-values to define equal_variances
    levene_stat, levene_pval = levene(*samples).statistic, levene(*samples).pvalue
    equal_variances = levene_pval > 0.05

    # save the info for return and text for output
    levene_info = {grouping_variable: {'stat': levene_stat, 'p': levene_pval, 'equal_variances': equal_variances}}
    levene_text = f"Results for samples in groups of '{grouping_variable}' for ['{target_variable}'] target variable:\n  ➡ statistic: {levene_info[grouping_variable]['stat']}\n  ➡ p-value: {levene_info[grouping_variable]['p']}\n{(f'  ∴ Equal variances: Yes (H0 cannot be rejected)' if levene_info[grouping_variable]['equal_variances'] else f'  ∴ Equal variances: No (H0 rejected)')}\n\n"

    # output & return
    print(f"< EQUAL VARIANCES TESTING: LEVENE >\n")
    print(levene_text)
    return levene_info


equal_variances_info = testing_equal_variances(test_df, variable, grouping)


#   , equal_variances_info
def perform_hypothesis_testing_numeric(df, target_variable, grouping_variable, normality_info, equal_variances_info):
    groups = [group for group in normality_info.keys()]
    samples = [df[df[grouping_variable] == group][target_variable] for group in groups]

    # unpack info on normality and equal variance
    normality_bool = all([info['normality'] for group, info in normality_info.items()])
    equal_variances_bool = equal_variances_info[grouping_variable]['equal_variances']

    # define output object
    output_info = []

    if len(samples) == 2:  # two sample testing
        if normality_bool and equal_variances_bool:  # parametric testing
            stat, p_val = ttest_ind(*samples)
            test_name = 'Independent Samples T-Test'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference between the group means, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference between the group means, rejecting the null hypothesis."
            test_info = {'stat': stat, 'p_val': p_val, 'test_name': test_name, 'conclusion': conclusion}
            output_info.append(test_info)
        else:  # non-parametric testing
            stat, p_val = mannwhitneyu(*samples)
            test_name = 'Mann-Whitney U Rank Test (Two Independent Samples)'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference in group distributions, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference in group distributions, rejecting the null hypothesis."
            test_info = {'stat': stat, 'p_val': p_val, 'test_name': test_name, 'conclusion': conclusion}
            output_info.append(test_info)
    else:  # more than two samples
        if normality_bool and equal_variances_bool:
            stat, p_val = f_oneway(*samples)
            test_name = f'One-way ANOVA (with {len(samples)} groups)'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference among the group means, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference among the group means, rejecting the null hypothesis."
            test_info = {'stat': stat, 'p_val': p_val, 'test_name': test_name, 'conclusion': conclusion}
            output_info.append(test_info)
        else:
            stat, p_val = kruskal(*samples)
            test_name = f'Kruskal-Wallis H-test (with {len(samples)} groups)'
            conclusion = "The data do not provide sufficient evidence to conclude a significant difference among the group distributions, failing to reject the null hypothesis." if p_val > 0.05 else "The data provide sufficient evidence to conclude a significant difference among the group distributions, rejecting the null hypothesis."
            test_info = {'stat': stat, 'p_val': p_val, 'test_name': test_name, 'conclusion': conclusion}
            output_info.append(test_info)

    # construct console output and return
    generate_test_names = (output_info[0]['test_name'] if len(output_info) == 1 else [info['test_name'] for info in output_info])
    print(f"< HYPOTHESIS TESTING: {generate_test_names}>\nBased on:\n  ➡ Normality assumption: {'✔' if normality_bool else '✘'}\n  ➡ Equal variances assumption: {'✔' if equal_variances_bool else '✘'}\n  ➡ Nr. of Groups: {len(samples)} groups\n  ∴ Performing {generate_test_names}:\n")
    [print(f"Results of {info['test_name']}:\n  ➡ statistic: {info['stat']}\n  ➡ p-value: {info['p_val']}\n  ∴ Conclusion: {info['conclusion']}\n") for info in output_info]
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
    output_info = []

    if chi2_bool:
        if yates_correction_shape_bool and sample_size <= yates_min_sample_size:
            stat, p_val, dof, expected_frequencies = chi2_contingency(contingency_table, correction=True)
            test_name = f"Chi-square test of independence of variables (with Yate's Correction)"
            chi2_tip = f"\n☻ Tip: The Chi-square test of independence with Yates' Correction is used for 2x2 contingency tables with small sample sizes. Yates' Correction makes the test more conservative, reducing the Type I error rate by adjusting for the continuity of the chi-squared distribution. This correction is typically applied when sample sizes are small (often suggested for total sample sizes less than about 40), aiming to avoid overestimation of statistical significance."
            conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f}, with Yate's Correction)." if p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f}, with Yate's Correction)."
            chi2_output_info = {'stat': stat, 'p_val': p_val, 'test_name': test_name, 'tip': chi2_tip, 'conclusion': conclusion}
            output_info.append(chi2_output_info)
        else:
            stat, p_val, dof, expected_frequencies = chi2_contingency(contingency_table, correction=False)
            test_name = f"Chi-square test of independence of variables (without Yate's Correction)"
            chi2_tip = f"\n☻ Tip: The Chi-square test of independence without Yates' Correction is preferred when analyzing larger contingency tables or when sample sizes are sufficiently large, even for 2x2 tables (often suggested for total sample sizes greater than 40). Removing Yates' Correction can increase the test's power by not artificially adjusting for continuity, making it more sensitive to detect genuine associations between variables in settings where the assumptions of the chi-squared test are met."
            conclusion = f"There is no statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f}, without Yate's Correction)." if p_val > 0.05 else f"There is a statistically significant association between {categorical_variable1} and {categorical_variable2} (p = {p_val:.3f}, without Yate's Correction)."
            chi2_output_info = {'stat': stat, 'p_val': p_val, 'test_name': test_name, 'tip': chi2_tip, 'conclusion': conclusion}
            output_info.append(chi2_output_info)
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
            bernard_output_info = {'stat': barnard_stat, 'p_val': barnard_p_val, 'test_name': bernard_test_name, 'tip': bernard_tip, 'conclusion': bernard_conclusion}
            output_info.append(bernard_output_info)

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
            boschloo_output_info = {'stat': boschloo_stat, 'p_val': boschloo_p_val, 'test_name': boschloo_test_name, 'tip': boschloo_tip, 'conclusion': boschloo_conclusion}
            output_info.append(boschloo_output_info)

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
            fisher_output_info = {'stat': fisher_stat, 'p_val': fisher_p_val, 'test_name': fisher_test_name, 'tip': fisher_tip, 'conclusion': fisher_conclusion}
            output_info.append(fisher_output_info)

    # console output & return
    generate_test_names = (output_info[0]['test_name'] if len(output_info) == 1 else [info['test_name'] for info in output_info])
    print(f"< HYPOTHESIS TESTING: {generate_test_names}>\nBased on:\n  ➡ Sample size: {sample_size}\n  ➡ Minimum Observed Frequency: {min_observed_frequency}\n  ➡ Minimum Expected Frequency: {min_expected_frequency}\n  ➡ Contingency table shape: {contingency_table_shape[0]}x{contingency_table_shape[1]}\n  ∴ Performing {generate_test_names}:\n")
    [print(f"Results of {info['test_name']}:\n  ➡ statistic: {info['stat']}\n  ➡ p-value: {info['p_val']}\n  ∴ Conclusion: {info['conclusion']}{info['tip']}\n") for info in output_info]
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
