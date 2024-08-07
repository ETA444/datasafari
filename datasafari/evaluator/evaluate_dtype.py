# DataSafari - DataSafari simplifies complex data science tasks into straightforward, powerful one-liners.
# Copyright (C) 2024 George Dreemer.
#
# Read more about DataSafari's LICENSE here: https://datasafari.dev/docs/other/lic-gpl3
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details, specifically under
# version 3 of the License. No later versions are applicable.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see: https://github.com/ETA444/datasafari/blob/main/LICENSE

import warnings
from typing import Union
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype


def evaluate_dtype(
        df: pd.DataFrame,
        col_names: list,
        max_unique_values_ratio: float = 0.05,
        min_unique_values: int = 10,
        string_length_threshold: int = 50,
        small_dataset_threshold: int = 20,
        output: str = 'dict'
) -> Union[dict, list]:
    """
    **Evaluate and automatically categorize the data types of DataFrame columns, effectively distinguishing between ambiguous cases based on detailed logical assessments.**

    This function examines columns within a DataFrame to determine if they are numerical or categorical. It goes beyond simple data type checks by considering the distribution of unique values within numerical data. This allows for more nuanced categorization, where numerical columns with a limited range of unique values can be treated as categorical, based on specified thresholds.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to be evaluated.

    col_names : list
        A list of column names whose data types are to be evaluated.

    max_unique_values_ratio : float, optional, default: 0.05
        The maximum ratio of unique values to total observations for a column to be considered categorical.

    min_unique_values : int, optional, default: 10
        The minimum number of unique values for a column to be considered continuous. Columns below this threshold are categorized.

    string_length_threshold : int, optional, default: 50
        The average string length threshold above which a column is classified as text data.

    small_dataset_threshold : int, optional, default: 20
        The threshold for small datasets, below which the column is likely categorical.

    output : str, optional, default: 'dict'
        Specifies the format of the output.
            - ``'dict'`` Returns a dictionary mapping column names to their determined data types ('numerical', 'categorical', 'text', 'datetime').
            - ``'list_n'`` Returns a list of booleans indicating whether each column in `col_names` is numerical (True) or not (False).
            - ``'list_c'`` Returns a list of booleans indicating whether each column in `col_names` is categorical (True) or not (False).
            - ``'list_d'`` Returns a list of booleans indicating whether each column in `col_names` is datetime (True) or not (False).
            - ``'list_t'`` Returns a list of booleans indicating whether each column in `col_names` is text (True) or not (False).

    Returns:
    --------
    dict or list
       Depending on the 'output' parameter, this function returns a dictionary or list.
           - ``dict`` If output='dict': A dictionary mapping column names to their determined data types ('numerical', 'categorical', 'text' or 'datetime').
           - ``list`` If output='list_n/c/d/t': A list of booleans indicating the nature of each column in `col_names` (in the same order), according to the specified 'output' parameter.

    Raises:
    -------
    TypeErrors:
        - If `df` is not a pandas DataFrame.
        - If `col_names` is not a list or if elements of `col_names` are not all strings.
        - If `max_unique_values_ratio` is not a float or an integer.
        - If `min_unique_values`, `string_length_threshold`, `small_dataset_threshold` are not integers.
        - If `output` is not a string or does not match one of the expected output format strings.

    ValueErrors:
        - If the `df` is empty.
        - If `max_unique_values_ratio` is outside the range [0, 1].
        - If `min_unique_values` is less than 1, as at least one unique value is needed to categorize a column.
        - If `string_length_threshold` is less than or equal to 0, indicating an invalid threshold for text data classification.
        - If 'col_names' list is empty or any specified column names in `col_names` are not present in the DataFrame.
        - If the `output` string does not correspond to one of the valid options.

    Examples:
    ---------
    Create an example DataFrame with mixed data types:

    >>> import datasafari
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {
    ...    'Age': np.random.randint(18, 35, 100),
    ...    'Income': np.random.normal(50000, 15000, 100),
    ...    'Department': np.random.choice(['HR', 'Tech', 'Admin'], 100),
    ...    'Entry Date': pd.date_range(start='2021-01-01', periods=100, freq='M')
    ... }
    >>> df = pd.DataFrame(data)

    Evaluating data types with a dictionary output format:

    >>> data_type_dict = evaluate_dtype(df, ['Age', 'Income', 'Department', 'Entry Date'], output='dict')
    >>> print(data_type_dict)

    Evaluating data types with a list output format indicating numerical data:

    >>> numerical_bool_list = evaluate_dtype(df, ['Age', 'Income', 'Department', 'Entry Date'], output='list_n')
    >>> print(numerical_bool_list)

    Evaluating data types with a list output format indicating categorical data:

    >>> categorical_bool_list = evaluate_dtype(df, ['Age', 'Income', 'Department', 'Entry Date'], output='list_c')
    >>> print(categorical_bool_list)

    Evaluating data types with a list output format indicating datetime data:

    >>> datetime_bool_list = evaluate_dtype(df, ['Age', 'Income', 'Department', 'Entry Date'], output='list_d')
    >>> print(datetime_bool_list)
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

    if not isinstance(string_length_threshold, int):
        raise TypeError("evaluate_dtype(): The 'string_length_threshold' must be an integer.")

    if not isinstance(small_dataset_threshold, int):
        raise TypeError("evaluate_dtype(): The 'small_dataset_threshold' must be an integer.")

    if not isinstance(output, str):
        raise TypeError("evaluate_dtype(): The 'output' parameter must be a string. Possible values are: 'dict', 'list_n', 'list_c', 'list_d' and 'list_t'.")

    # ValueErrors
    if df.empty:
        raise ValueError("evaluate_dtype(): The input DataFrame is empty.")

    if max_unique_values_ratio < 0 or max_unique_values_ratio > 1:
        raise ValueError("evaluate_dtype(): The 'max_unique_values_ratio' must be between 0 and 1.")

    if min_unique_values < 1:
        raise ValueError("evaluate_dtype(): The 'min_unique_values' must be at least 1.")

    if string_length_threshold <= 0:
        raise ValueError("evaluate_dtype(): The 'string_length_threshold' must be greater than 0.")

    if small_dataset_threshold <= 0:
        raise ValueError("evaluate_dtype(): The 'small_dataset_threshold' must be greater than 0.")

    if len(col_names) == 0:
        raise ValueError("evaluate_dtype(): The 'col_names' list must contain at least one column name.")

    missing_cols = [col for col in col_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"evaluate_dtype(): The following columns were not found in the DataFrame: {', '.join(missing_cols)}")

    valid_outputs = ['dict', 'list_n', 'list_c', 'list_d', 'list_t']
    if output not in valid_outputs:
        raise ValueError(f"evaluate_dtype(): Invalid output '{output}'. Valid options are: {', '.join(valid_outputs)}")

    # Warn about small dataset size #
    if df.shape[0] < 60:
        warnings.warn(
            f"evaluate_dtype: Dataset size ({df.shape[0]}) is smaller than 60. "
            "Evaluation may be inaccurate.",
            UserWarning
        )

    # Main Function #
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
            num_unique_values = df[col].nunique()
            total_values = len(df[col])
            if total_values <= small_dataset_threshold and num_unique_values < min_unique_values:
                data_type_dictionary[col] = 'categorical'
            elif avg_str_length > string_length_threshold or num_unique_values / total_values > max_unique_values_ratio:
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
        categorical_type_list = [dtype == 'categorical' for dtype in data_type_dictionary.values()]
        return categorical_type_list
    elif output.lower() == 'list_d':
        datetime_type_list = [dtype == 'datetime' for dtype in data_type_dictionary.values()]
        return datetime_type_list
    elif output.lower() == 'list_t':
        text_type_list = [dtype == 'text' for dtype in data_type_dictionary.values()]
        return text_type_list
