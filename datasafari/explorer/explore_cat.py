import pandas as pd
from datasafari.utils import calculate_entropy
from datasafari.evaluator import evaluate_dtype


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

    Raises
    ------
    TypeError
        - If `df` is not a pandas DataFrame.
        - If `categorical_variables` is not a list or contains non-string elements.
        - If `method` or `output` is not a string.
    ValueError
        - If `method` is not one of the valid options ('unique_values', 'counts_percentage', 'entropy', 'all').
        - If `output` is not one of the valid options ('print', 'return').
        - If 'categorical_variables' list is empty.
        - If variables provided through 'categorical_variables' are not categorical variables.
        - If any of the specified categorical variables are not found in the DataFrame.

    Examples:
    --------
    # Create a sample DataFrame to use in the examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> data = {
    ...     'Category1': np.random.choice(['Apple', 'Banana', 'Cherry'], size=100),
    ...     'Category2': np.random.choice(['Yes', 'No'], size=100),
    ...     'Category3': np.random.choice(['Low', 'Medium', 'High'], size=100)
    ... }
    >>> df = pd.DataFrame(data)

    # Display unique values for 'Category1' and 'Category2'
    >>> explore_cat(df, ['Category1', 'Category2'], method='unique_values', output='print')

    # Explore counts and percentages for 'Category1' and 'Category2', then print the results
    >>> explore_cat(df, ['Category1', 'Category2'], method='counts_percentage', output='print')

    # Calculate and return the entropy of 'Category1', 'Category2', and 'Category3'
    >>> result = explore_cat(df, ['Category1', 'Category2', 'Category3'], method='entropy', output='return')
    >>> print(result)

    # Comprehensive exploration of all specified methods for 'Category1', 'Category2', and 'Category3', displaying to console
    >>> explore_cat(df, ['Category1', 'Category2', 'Category3'], method='all', output='print')

    # Using 'all' method to explore 'Category1' and 'Category2', returning the results as a string
    >>> result_str = explore_cat(df, ['Category1', 'Category2'], method='all', output='return')
    >>> print(result_str)

    Notes:
    -----
    The 'entropy' method provides a quantitative measure of the unpredictability or
    diversity within each specified categorical column, calculated as outlined in the
    documentation for 'calculate_entropy'. High entropy values indicate a more uniform
    distribution of categories, suggesting no single category overwhelmingly dominates.
    """

    # Error Handling #
    # TypeErrors
    if not isinstance(df, pd.DataFrame):
        raise TypeError("explore_cat(): The df parameter must be a pandas DataFrame.")

    if not isinstance(categorical_variables, list):
        raise TypeError(f"explore_cat(): The categorical_variables parameter must be a list of variable names.\n Example: var_list = ['var1', 'var2', 'var3']")
    else:
        if not all(isinstance(var, str) for var in categorical_variables):
            raise TypeError("explore_cat(): All items in the categorical_variables list must be strings representing column names.")

    if not isinstance(method, str):
        raise TypeError(f"explore_cat(): The method parameter must be a string.\n Example: method = 'all'")

    if not isinstance(output, str):
        raise TypeError(f"explore_cat(): The output parameter must be a string. \n Example: output = 'return'")

    # ValueErrors

    # Check if method is valid
    valid_methods = ['unique_values', 'counts_percentage', 'entropy', 'all']
    if method.lower() not in valid_methods:
        raise ValueError(f"explore_cat(): Invalid method '{method}'. Valid options are: {', '.join(valid_methods)}")

    # Check if output is valid
    if output.lower() not in ['print', 'return']:
        raise ValueError("explore_cat(): Invalid output method. Choose 'print' or 'return'.")

    # Check if list has any members
    if len(categorical_variables) == 0:
        raise ValueError("explore_cat(): The 'categorical_variables' list must contain at least one column name.")

    # Check if variables are categorical
    categorical_types = evaluate_dtype(df, categorical_variables, output='list_c')
    if not all(categorical_types):
        raise ValueError(f"explore_cat(): The 'categorical_variables' list must contain only names of categorical variables.")

    # Check if specified variables exist in the DataFrame
    missing_vars = [var for var in categorical_variables if var not in df.columns]
    if missing_vars:
        raise ValueError(f"explore_cat(): The following variables were not found in the DataFrame: {', '.join(missing_vars)}")

    # Main Function #
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
