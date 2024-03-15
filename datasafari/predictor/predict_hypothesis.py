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


def test_numeric_assumptions(df, grouping_variable, value_variable):
    groups = df[grouping_variable].unique().tolist()
    # normality
    shapiro_pvals = [shapiro(df[df[grouping_variable] == group][value_variable]).pvalue for group in groups]
    shapiro_stats = [shapiro(df[df[grouping_variable] == group][value_variable]).statistic for group in groups]
    shapiro_info = {group: [shapiro_pvals[n], shapiro_stats[n]] for n, group in enumerate(groups)}
    shapiro_text = [f"Shapiro Results for {key} group in variable {value_variable}:\n- statistic: {value[1]}\n- p-value: {value[0]}\n" for key, value in shapiro_info.items()]

    [print(text) for text in shapiro_text]


test_numeric_assumptions(test_df, grouping, variable)
