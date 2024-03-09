import pandas as pd
import numpy as np
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
    # DONE! TODO: Add console output for standardize

    if method == 'standardize':
        print(f"< STANDARDIZING DATA >")
        print(f" This method centers the data around mean 0 with a standard deviation of 1, enhancing model performance and stability.")
        print(f"  ✔ Standardizes each numerical variable to have mean=0 and variance=1.")
        print(f"  ✔ Essential preprocessing step for many machine learning algorithms.\n")
        print(f"✎ Note: Standardization is applied only to the specified numerical variables.\n")

        # initialize essential objects
        transformed_df = df.copy()
        scaler = StandardScaler()

        # scale the data
        transformed_df[numerical_variables] = scaler.fit_transform(df[numerical_variables])

        # isolate transformed columns to give as part of output
        standardized_columns = transformed_df[numerical_variables]

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the scaled columns:\n{standardized_columns.head()}\n")
        print("☻ HOW TO - to catch the dfs: `transformed_df, standardized_columns = transform_num(your_df, your_numerical_variables, method='standardize')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")

        return transformed_df, standardized_columns

    if method.lower() == 'log':
        print(f"< LOG TRANSFORMATION >")
        print(f" This method applies a natural logarithm transformation to positively skewed data.")
        print(f"  ✔ Helps to stabilize variance and make the data more normally distributed.")
        print(f"  ✔ Particularly useful for data with a heavy right tail (positively skewed).\n")
        print(f"✎ Note: Log transformation is applied only to specified numerical variables. Zero or negative values in the data can cause issues.\n")

        # initialize the DataFrame to work with
        transformed_df = df.copy()

        # check for and handle zero or negative values to avoid log transformation issues
        if (transformed_df[numerical_variables] <= 0).any().any():
            print(f"⚠️ Warning: Zero or negative values detected. Incrementing by 1 to avoid log(0) and negative log issues.\n")
            transformed_df[numerical_variables] += 1

        # apply log transformation
        transformed_df[numerical_variables] = np.log(transformed_df[numerical_variables])

        # isolate transformed columns to give as part of output
        log_transformed_columns = transformed_df[numerical_variables]

        print(f"✔ Numerical variables have been log-transformed.\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, log_transformed_columns = transform_num(your_df, your_numerical_variables, method='log')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")
        print("* Consider examining the distribution of your data post-transformation.\n")

        return transformed_df, log_transformed_columns


# smoke testing #



# standardize
