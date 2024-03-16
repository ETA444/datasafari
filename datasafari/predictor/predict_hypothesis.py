import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene


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
    'Feature1': np.random.normal(0, 1, 100),
    'Feature2': np.random.exponential(1, 100),
    'Feature3': np.random.randint(1, 100, 100)
}
test_df = pd.DataFrame(data)
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


def test_normality(df, grouping_variable, target_variable):
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


test_normality(test_df, grouping, variable)


def test_equal_variances(df, grouping_variable, target_variable):
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


test_equal_variances(test_df, grouping, variable)
