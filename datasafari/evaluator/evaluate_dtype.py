import pandas as pd


def evaluate_dtype(df: pd.DataFrame, col_names: list, max_unique_values_ratio: float = 0.05, min_unique_values: int = 10, output: str = 'dict'):

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
