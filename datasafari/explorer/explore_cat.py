import pandas as pd
from datasafari.utils import calculate_entropy


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
