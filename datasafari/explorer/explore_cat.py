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

    if method.lower() in ['unique', 'all']:
        # initial append for title of method section
        result.append(f"<<______UNIQUE VALUES PER VARIABLE______>>\n")

        # get the unique values per variable in categorical_variables list
        for variable_name in categorical_variables:
            unique_values = df[variable_name].unique()
            result.append(f"['{variable_name}'] Unique values:\n{unique_values}")

    # Combine all results
    combined_result = "\n".join(result)

    if output.lower() == 'print':
        print(combined_result)
    elif output.lower() == 'return':
        return combined_result
    else:
        raise ValueError("Invalid output method. Choose 'print' or 'return'.")


# smoke tests
test_df = pd.read_csv('./data/soil_measures.csv')
variables_of_interest = ['crop']
unique = explore_cat(
    df=test_df,
    categorical_variables=variables_of_interest,
    method='unique',
    output='return'
)

test_df2 = pd.read_csv('./data/rental_info.csv')
print(test_df2.info())
explore_cat(test_df2, ['special_features', 'release_year'])
