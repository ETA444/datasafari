import pandas as pd
import numpy as np


# main function: explore_cat
def explore_cat(
        df: pd.DataFrame,
        categorical_variables: list,
        method: str = 'all',
        output: str = 'print'):
    # docstring here
    result = []

    if method.lower() in ['unique_values', 'all']:
        # initial append for title of method section
        result.append(f"<<______UNIQUE VALUES PER VARIABLE______>>\n")

        # get the unique values per variable in categorical_variables list
        for variable_name in categorical_variables:
            unique_values = df[variable_name].unique()
            result.append(f"< Unique values of ['{variable_name}'] >\n\n{unique_values}\n\n")

    if method.lower() in ['counts_percentage', 'all']:
        # initial append for title of method section
        result.append(f"<<______COUNTS & PERCENTAGE______>>\n")

        # get the counts and percentages per unique value of variable in categorical_variables list
        for variable_name in categorical_variables:
            counts = df[variable_name].value_counts()
            percentages = df[variable_name].value_counts(normalize=True) * 100

            # combine counts and percentages into a DataFrame
            summary_df = pd.DataFrame({'Counts': counts, 'Percentages': percentages})

            # format percentages to be 2 decimals
            summary_df['Percentages'] = summary_df['Percentages'].apply(lambda x: f"{x:.2f}%")

            result.append(f"< Counts and percentages per unique value of ['{variable_name}'] >\n\n{summary_df}\n\n")

    # Combine all results
    combined_result = "\n".join(result)

    if output.lower() == 'print':
        print(combined_result)
    elif output.lower() == 'return':
        return combined_result
    else:
        raise ValueError("Invalid output method. Choose 'print' or 'return'.")


# smoke tests
df1 = pd.read_csv('./datasets/soil_measures.csv')
list1 = ['crop']
unique = explore_cat(
    df=df1,
    categorical_variables=list1,
    method='unique',
    output='return'
)

df2 = pd.read_csv('./datasets/rental_info.csv')
list2 = ['special_features', 'release_year']

explore_cat(df2, list2)
