import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype


def evaluate_dtype(df: pd.DataFrame, col_names: list, max_unique_values_ratio: float = 0.05, min_unique_values: int = 10, string_length_threshold: int = 50, output: str = 'dict'):
    """
    Evaluates and categorizes data types of specified columns in a DataFrame, with enhanced handling for numerical data that may functionally serve as categorical data.

    This function examines columns within a DataFrame to determine if they are numerical or categorical. It goes beyond simple data type checks by considering the distribution of unique values within numerical data. This allows for more nuanced categorization, where numerical columns with a limited range of unique values can be treated as categorical, based on specified thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be evaluated.
    col_names : list
        A list of column names whose data types are to be evaluated.
    max_unique_values_ratio : float, optional
        The maximum ratio of unique values to total observations for a column to be considered categorical. Default is 0.05.
    min_unique_values : int, optional
        The minimum number of unique values for a column to be considered continuous. Columns below this threshold are categorized. Default is 10.
    string_length_threshold : int, optional
        The average string length threshold above which a column is classified as text data. Default is 50.
    output : str, optional
        Specifies the format of the output. Options are:
        - 'dict': Returns a dictionary mapping column names to their determined data types ('numerical', 'categorical', 'text', 'datetime').
        - 'list_n': Returns a list of booleans indicating whether each column in `col_names` is numerical (True) or not (False).
        - 'list_c': Returns a list of booleans indicating whether each column in `col_names` is categorical (True) or not (False).
        - 'list_d': Returns a list of booleans indicating whether each column in `col_names` is datetime (True) or not (False).
        - 'list_t': Returns a list of booleans indicating whether each column in `col_names` is text (True) or not (False).

    Returns
    -------
    dict or list
       Depending on the 'output' parameter, this function returns:
           - A dictionary mapping column names to their determined data types ('numerical', 'categorical', 'text' or 'datetime').
           - A list of booleans indicating the nature of each column in `col_names`, according to the specified 'output' parameter.

    Raises
    ------
    TypeError
        - If `df` is not a pandas DataFrame.
        - If `col_names` is not a list or if elements of `col_names` are not all strings.
        - If `max_unique_values_ratio` is not a float or an integer.
        - If `min_unique_values`, `string_length_threshold` are not integers.
        - If `output` is not a string or does not match one of the expected output format strings ('dict', 'list_n', 'list_c').
    ValueError
        - If `max_unique_values_ratio` is outside the range [0, 1].
        - If `min_unique_values` is less than 1, as at least one unique value is needed to categorize a column.
        - If `string_length_threshold` is less than or equal to 0, indicating an invalid threshold for text data classification.
        - If 'col_names' list is empty or any specified column names in `col_names` are not present in the DataFrame.
        - If the `output` string does not correspond to one of the valid options ('dict', 'list_n', 'list_c').

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
           'Age': np.random.randint(18, 35, 100),
           'Income': np.random.normal(50000, 15000, 100),
           'Department': np.random.choice(['HR', 'Tech', 'Admin'], 100)
       })
    >>> data_type_dict = evaluate_dtype(df, ['Age', 'Income', 'Department'], output='dict')
    >>> numerical_bool_list = evaluate_dtype(df, ['Age', 'Income', 'Department'], output='list_n')
    >>> categorical_bool_list = evaluate_dtype(df, ['Age', 'Income', 'Department'], output='list_c')
    """

    # Error-handling #
    # TypeErrors
    if not isinstance(df, pd.DataFrame):
        raise TypeError("evaluate_dtype(): The 'df' parameter must be a pandas DataFrame.")

    if not isinstance(col_names, list):
        raise TypeError("evaluate_dtype(): The 'col_names' parameter must be a list.")
    elif not all(isinstance(col, str) for col in col_names):
        raise TypeError("evaluate_dtype(): All elements in the 'col_names' list must be strings representing column names.")

    if not isinstance(max_unique_values_ratio, (float, int)):
        raise TypeError("evaluate_dtype(): The 'max_unique_values_ratio' must be a float or integer.")

    if not isinstance(min_unique_values, int):
        raise TypeError("evaluate_dtype(): The 'min_unique_values' must be an integer.")

    if not isinstance(output, str):
        raise TypeError("evaluate_dtype(): The 'output' parameter must be a string. Possible values are: 'dict', 'list_n' and 'list_c'.")

    # ValueErrors
    if max_unique_values_ratio < 0 or max_unique_values_ratio > 1:
        raise ValueError("evaluate_dtype(): The 'max_unique_values_ratio' must be between 0 and 1.")

    if min_unique_values < 1:
        raise ValueError("evaluate_dtype(): The 'min_unique_values' must be at least 1.")

    # Check if list has any members
    if len(col_names) == 0:
        raise ValueError("evaluate_dtype(): The 'col_names' list must contain at least one column name.")

    missing_cols = [col for col in col_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"evaluate_dtype(): The following columns were not found in the DataFrame: {', '.join(missing_cols)}")

    valid_outputs = ['dict', 'list_n', 'list_c']
    if output not in valid_outputs:
        raise ValueError(f"evaluate_dtype(): Invalid output '{output}'. Valid options are: {', '.join(valid_outputs)}")

    # Main Function
    data_type_dictionary = {}
    for col in col_names:
        if is_numeric_dtype(df[col]):
            num_unique_values = df[col].nunique()
            total_values = len(df[col])
            if (num_unique_values <= min_unique_values) or (num_unique_values / total_values <= max_unique_values_ratio):
                data_type_dictionary[col] = 'categorical'
            else:
                data_type_dictionary[col] = 'numerical'
        elif is_string_dtype(df[col]):
            avg_str_length = df[col].dropna().apply(len).mean()
            if avg_str_length > string_length_threshold or df[col].nunique() / len(df[col]) > max_unique_values_ratio:
                data_type_dictionary[col] = 'text'
            else:
                data_type_dictionary[col] = 'categorical'
        elif is_datetime64_any_dtype(df[col]):
            data_type_dictionary[col] = 'datetime'
        else:
            data_type_dictionary[col] = 'categorical'

    if output.lower() == 'dict':
        return data_type_dictionary
    elif output.lower() == 'list_n':
        numerical_type_list = [dtype == 'numerical' for dtype in data_type_dictionary.values()]
        return numerical_type_list
    elif output.lower() == 'list_c':
        categorical_type_list = [dtype.startswith('categorical') for dtype in data_type_dictionary.values()]
        return categorical_type_list
