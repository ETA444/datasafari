import pandas as pd


def evaluate_dtype(df: pd.DataFrame, col_names: list, max_unique_values_ratio: float = 0.05, min_unique_values: int = 10, output: str = 'dict'):

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

    missing_cols = [col for col in col_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"evaluate_dtype(): The following columns were not found in the DataFrame: {', '.join(missing_cols)}")

    valid_outputs = ['dict', 'list_n', 'list_c']
    if output not in valid_outputs:
        raise ValueError(f"evaluate_dtype(): Invalid output '{output}'. Valid options are: {', '.join(valid_outputs)}")

    data_type_dictionary = {}
    numerical_type_list = []
    categorical_type_list = []
    for col in col_names:

        if df[col].dtype.name == 'category':
            data_type_dictionary[col] = 'categorical'
            numerical_type_list.append(False)
            categorical_type_list.append(True)

        elif df[col].dtype.kind in 'iuf':  # Integer, unsigned integer, float
            num_unique_values = df[col].nunique()
            total_values = len(df[col])

            if (num_unique_values <= min_unique_values) or (num_unique_values / total_values <= max_unique_values_ratio):
                data_type_dictionary[col] = 'categorical'
                numerical_type_list.append(False)
                categorical_type_list.append(True)
            else:
                data_type_dictionary[col] = 'numerical'
                numerical_type_list.append(True)
                categorical_type_list.append(False)
        else:
            # Default to categorical for other data types (e.g., string objects)
            data_type_dictionary[col] = 'categorical'
            numerical_type_list.append(False)
            categorical_type_list.append(True)

    if output.lower() == 'dict':
        return data_type_dictionary
    elif output.lower() == 'list_n':
        return numerical_type_list
    elif output.lower() == 'list_c':
        return categorical_type_list
