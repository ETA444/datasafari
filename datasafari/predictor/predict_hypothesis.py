import numpy as np
import pandas as pd
from scipy.stats import (
    shapiro, levene,  # numerical assumptions
    ttest_ind, mannwhitneyu, f_oneway, kruskal,  # numerical testing
    chi2_contingency, barnard_exact, boschloo_exact, fisher_exact  # categorical testing
)
from scipy.stats.contingency import expected_freq


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


# TODO: Implement test preference mechanism where users can choose more tests (e.g. ...)
def testing_normality(df, target_variable, grouping_variable, test: str = 'shapiro'):
    groups = df[grouping_variable].unique().tolist()

    # shapiro-wilk test #
    # calculating statistic and p-values to define normality
    shapiro_stats = [shapiro(df[df[grouping_variable] == group][target_variable]).statistic for group in groups]
    shapiro_pvals = [shapiro(df[df[grouping_variable] == group][target_variable]).pvalue for group in groups]
    normality = [True if p > 0.05 else False for p in shapiro_pvals]

    # save the info for return and text for output
    shapiro_info = {group: {'stat': shapiro_stats[n], 'p': shapiro_pvals[n], 'normality': normality[n]} for n, group in enumerate(groups)}
    shapiro_text = [f"Results for '{key}' group in variable ['{target_variable}']:\n  ➡ statistic: {value['stat']}\n  ➡ p-value: {value['p']}\n{(f'  ∴ Normality: Yes (H0 cannot be rejected)' if value['normality'] else f'  ∴ Normality: No (H0 rejected)')}\n\n" for key, value in shapiro_info.items()]

    # output & return
    print(f"< NORMALITY TESTING: SHAPIRO-WILK >\n")
    print(*shapiro_text)
    return shapiro_info


normality_info = testing_normality(test_df, variable, grouping)


# TODO: Implement test preference mechanism where users can choose more tests (e.g. bartlett)
def testing_equal_variances(df, target_variable, grouping_variable, test: str = 'shapiro'):
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

