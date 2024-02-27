import pandas as pd


# main function: transform_cat
def transform_cat(df: pd.DataFrame, categorical_variables: list, method: str = 'all', output: str = 'print'):
    print('hi time for todo ideas')
    # TODO: preprocess categorical data to make: uniform (e.g. Education column has University and university, etc.)
    # TODO: easily create ordinal categorical variables
    # TODO: easily create nominal categorical variables
    # TODO: ordinal encoding with overwrite or no overwrite
    # TODO: one hot encoding


# smoke tests
pengu = pd.read_csv('./datasets/penguins.csv')
