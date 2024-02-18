import pandas as pd
import numpy as np


# utility function: calculate_entropy
def calculate_entropy(series: pd.Series) -> float:
    """
    Calculate the entropy of a pandas Series (categorical variable), which
    quantifies the unpredictability or diversity of the data within the variable.

    Entropy is higher when the distribution of categories is more uniform, indicating
    a higher degree of randomness or diversity. Conversely, entropy is lower when
    one or a few categories dominate the distribution.

    The entropy (H) is calculated using the formula:

        H = -sum(p_i * log2(p_i))

    where p_i is the proportion of the series belonging to the ith category.

    Parameters:
    ----------
    series : pd.Series
        The series for which to calculate entropy. Should be a categorical variable
        with discrete values.

    Returns:
    -------
    float
        The entropy of the series. A non-negative value where higher values indicate
        greater entropy (diversity) and values close to zero indicate less entropy (more uniformity).

    Example:
    --------
    >>> example_series = pd.Series(['apple', 'orange', 'apple', 'banana', 'orange', 'banana'])
    >>> calculate_entropy(example_series)
    >>> 1.584962500721156  # Example entropy value for a balanced series
    """
    counts = series.value_counts()
    probabilities = counts / len(series)
    entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))  # adding epsilon to avoid log(0)
    return entropy


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

    if method.lower() in ['entropy', 'all']:
        result.append("<<______ENTROPY OF CATEGORICAL VARIABLES______>>\n")
        for variable_name in categorical_variables:
            entropy_val = calculate_entropy(df[variable_name])
            result.append(f"Entropy of ['{variable_name}']: {entropy_val:.4f}\n")

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
