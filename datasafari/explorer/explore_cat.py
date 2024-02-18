import pandas as pd
import numpy as np


# utility function: calculate_entropy
def calculate_entropy(series: pd.Series) -> (float, str):
    """
    Calculate the entropy of a pandas Series (categorical variable) and provide an interpretation by comparing it to
    the maximum possible entropy for the given number of unique categories. This function quantifies the
    unpredictability or diversity of the data within the variable.

    **Calculation:**
        - Entropy is computed using the formula: *H = -sum(p_i * log2(p_i))*
        - Where p_i is the proportion of the series belonging to the ith category. Higher entropy indicates a more uniform distribution of categories, suggesting a higher degree of randomness or diversity. Conversely, entropy is lower when one or a few categories dominate the distribution.

    **Interpretation:**
        The function also calculates the percentage of the maximum possible entropy (based on the number
        of unique categories) achieved by the actual entropy, providing context for how diverse the
        categorical data is relative to the total number of categories.

    Parameters:
    ----------
    series : pd.Series
        The series for which to calculate entropy. Should be a categorical variable with discrete values.

    Returns:
    -------
    tuple
        A tuple containing two elements:
            - A float representing the entropy of the series. Higher values indicate greater entropy (diversity), and values close to zero suggest less entropy (more uniformity).
            - A string providing an interpretation of the entropy value in the context of the maximum possible entropy for the given number of unique categories.

    Example:
    --------
    >>> example_series = pd.Series(['apple', 'orange', 'apple', 'banana', 'orange', 'banana'])
    >>> entropy_val_example, interpretation_example = calculate_entropy(example_series)
    >>> print(entropy_val_example)
    1.583
    >>> print(interpretation_example)
    "=> Moderate diversity [max. entropy for this variable = 3.161]"
    """
    counts = series.value_counts()
    probabilities = counts / len(series)
    entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))  # Adding epsilon to avoid log(0)

    # calculate maximum possible entropy for the number of unique categories
    unique_categories = len(counts)
    max_entropy = np.log2(unique_categories)

    # calculate entropy ratio which reflect level of diversity
    entropy_ratio = entropy / max_entropy

    # calculate the percentage of the maximum possible entropy - old
    # percent_of_max = (entropy / max_entropy) * 100
    # interpretation = f"{entropy:.4f} ({percent_of_max:.2f}% of max for {unique_categories} categories)"

    # build interpretation
    if entropy_ratio > 0.85:
        interpretation = f"\n=> High diversity [max. entropy for this variable = {max_entropy:.3f}]"
    elif entropy_ratio > 0.5:
        interpretation = f"\n=> Moderate diversity [max. entropy for this variable = {max_entropy:.3f}]"
    else:
        interpretation = f"\n=> Low diversity [max. entropy for this variable = {max_entropy:.3f}]"

    return entropy, interpretation


# main function: explore_cat
def explore_cat(df: pd.DataFrame, categorical_variables: list, method: str = 'all', output: str = 'print'):
    """
    Explores categorical variables within a DataFrame, providing insights through various methods. The exploration
    can yield unique values, counts and percentages of those values, and the entropy to quantify data diversity.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be explored.

    categorical_variables : list
        A list of strings specifying the names of the categorical columns to explore.

    method : str, default 'all'
        Specifies the method of exploration to apply. Options include:
            - 'unique_values': Lists unique values for each specified categorical variable.
            - 'counts_percentage': Shows counts and percentages for the unique values of each variable.
            - 'entropy': Calculates the entropy for each variable, providing a measure of data diversity. See the 'calculate_entropy' function for more details on entropy calculation.
            - 'all': Applies all the above methods sequentially.

    output : str, default 'print'
        Determines how the exploration results are outputted. Options are:
            - 'print': Prints the results to the console.
            - 'return': Returns the results as a single formatted string.

    Returns:
    --------
    str or None
        - If output='return', a string containing the formatted exploration results is returned.
        - If output='print', results are printed to the console, and the function returns None.

    Examples:
    --------
    # Explore all specified methods for 'Category1' and 'Category2' in 'df'
    >>> explore_cat(df, ['Category1', 'Category2'])

    # Explore with a focus on unique value counts and percentages, then print the results
    >>> explore_cat(df, ['Category1'], method='counts_percentage')

    # Calculate and return the entropy of 'Category1' and 'Category2'
    >>> result1 = explore_cat(df, ['Category1', 'Category2'], method='entropy', output='return')

    Notes:
    -----
    The 'entropy' method provides a quantitative measure of the unpredictability or
    diversity within each specified categorical column, calculated as outlined in the
    documentation for 'calculate_entropy'. High entropy values indicate a more uniform
    distribution of categories, suggesting no single category overwhelmingly dominates.
    """
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

        # include a tip on interpretation
        result.append("Tip: Higher entropy indicates greater diversity.*\n")

        for variable_name in categorical_variables:
            entropy_val, interpretation = calculate_entropy(df[variable_name])
            result.append(f"Entropy of ['{variable_name}']: {entropy_val:.3f} {interpretation}\n")

        # include additional info tip
        result.append("* For more details on entropy, run: 'print(calculate_entropy.__doc__)'.\n")

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
