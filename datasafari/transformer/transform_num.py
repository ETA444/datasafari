import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
        print("☻ HOW TO - Apply this transformation using `transformed_df, standardized_columns = transform_num(your_df, your_numerical_variables, method='standardize')`.\n")

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
        print(f"✎ Note: Log transformation is applied only to specified numerical variables. Zero or negative values in the data can cause issues and will skip those columns.\n")

        # initialize essential objects
        transformed_df = df.copy()
        log_transformed_columns = pd.DataFrame()
        skipped_columns = []

        # do the operation only after checking provided column for zero or negative values
        for variable in numerical_variables:
            if (transformed_df[variable] <= 0).any():
                print(f"⚠️ Warning: '{variable}' contains zero or negative values and was skipped to avoid log(0) and negative log issues.\n")
                skipped_columns.append(variable)
            else:
                transformed_column = np.log(transformed_df[variable])
                transformed_df[variable] = transformed_column
                log_transformed_columns = pd.concat([log_transformed_columns, transformed_column], axis=1)
                print(f"✔ '{variable}' has been log-transformed.\n")

        # inform user if a column was skipped
        if skipped_columns:
            print(f"Skipped columns due to non-positive values: {skipped_columns}\n")

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the scaled columns:\n{log_transformed_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, log_transformed_columns = transform_num(your_df, your_numerical_variables, method='log')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")
        print("* Consider examining the distribution of your data post-transformation.\n")

        return transformed_df, log_transformed_columns

    if method.lower() == 'normalize':
        print(f"< NORMALIZATION TRANSFORMATION >")
        print(f" This method scales numerical variables to a [0, 1] range, making them suitable for models sensitive to variable scales.")
        print(f"  ✔ Adjusts each feature to a [0, 1] scale based on its minimum and maximum values.")
        print(f"  ✔ Enhances model performance by ensuring numerical variables are on a similar scale.")
        print(f"✎ Note: Ensure data is clean and outliers are handled for optimal results.\n")
        print(f"☻ Tip: Use `explore_num()` for data inspection and outlier detection before applying normalization.\n")

        # initialize essentials
        transformed_df = df.copy()
        min_max_scaler = MinMaxScaler()

        # apply minmax scaling
        transformed_df[numerical_variables] = min_max_scaler.fit_transform(df[numerical_variables])

        # isolate transformed columns to provide as part of output
        normalized_columns = transformed_df[numerical_variables]

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the scaled columns:\n{normalized_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, normalized_columns = transform_num(your_df, your_numerical_variables, method='normalize')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")

        return transformed_df, normalized_columns


# smoke testing #

# example data to run tests on
data = {
    'Feature1': np.random.normal(0, 1, 100),  # Normally distributed data
    'Feature2': np.random.exponential(1, 100),  # Exponentially distributed data (positively skewed)
    'Feature3': np.random.randint(1, 100, 100)  # Uniformly distributed data between 1 and 100
}

df = pd.DataFrame(data)

num_cols = ['Feature1', 'Feature2', 'Feature3']

# standardize
standardized_data, standardized_cols = transform_num(df, num_cols, method='standardize')

# log
log_data, log_cols = transform_num(df, num_cols, method='log')

# normalize
normalized_data, normalized_cols = transform_num(df, num_cols, method='normalize')
