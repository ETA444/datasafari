import pandas as pd
from sklearn.preprocessing import StandardScaler


def transform_num(df, numerical_variables, method, **kwargs):
    """
    Apply various transformations to numerical variables in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the numerical data to be transformed.
    numerical_variables : list
        List of column names in `df` that are numerical and need transformation.
    method : str, optional
        The transformation method to apply. Default is 'standardize'.
        Supported methods: ['standardize', 'other_methods_here']
    **kwargs : dict
        Additional keyword arguments for different transformation methods.

    Returns
    -------
    pd.DataFrame
        A DataFrame with transformed numerical variables.

    Examples
    --------
    >>> df = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50]
        })
    >>> transformed_df = transform_num(df, ['Feature1', 'Feature2'], method='standardize')
    >>> print(transformed_df)
    """

    # transform_num TO DO #
    # TODO: Add console output for standardize

    if method == 'standardize':

        # initialize essential objects
        transformed_df = df.copy()
        scaler = StandardScaler()

        # scale the data
        transformed_df[numerical_variables] = scaler.fit_transform(df[numerical_variables])

        # isolate transformed columns to give as part of output
        standardized_columns = transformed_df[numerical_variables]

        return transformed_df, standardized_columns


# smoke testing #

# standardize
