import pandas as pd
import io
from datasafari.utils import filter_kwargs


# dictionary for filter_kwargs: define which kwargs are valid for which method
valid_kwargs = {
    'head': ['n'],
    'describe': ['percentiles', 'include', 'exclude'],
    'info': ['verbose', 'max_cols', 'memory_usage', 'show_counts']
}


# main function: explore_df
def explore_df(df: pd.DataFrame, method: str = 'all', output: str = 'print', **kwargs):
    """
    Explores a DataFrame using specified methods, with options for output control
    and method-specific parameters. The goal is to give you bird's eye view on your data.

    Note: For in-depth exploration of numerical and categorical data, we recommend using:
    - explore_num()
    - explore_cat()

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to explore.
    method : str, default 'all'
        The exploration method to apply on the DataFrame. Options include:
        - 'na': Display counts of NAs per column and percentage of NAs per column.
        - 'desc': Display summary statistics using describe().
        - 'head': Display the first few rows using head().
        - 'info': Display concise information about the DataFrame using info().
        - 'na': Display counts of NAs per column and percentage of NAs per column.
        - 'all': Execute all the above methods sequentially.
    output : str, default 'print'
        Controls the output of the exploration results.
        - Use 'print' to print the results to the console
        - Use 'return' to return the results as a string.
    **kwargs
        Additional arguments for pandas methods (e.g., 'percentiles' for 'desc').
        You can use 'all' and provide additional arguments that will be automatically
        matched to the appropriate pandas' method.
        - Note: .info(buf=...) is already occupied, thus disabled for use.

    Returns
    -------
    str or None
        String if output='return'. Otherwise, prints to console and returns None.

    Raises
    ------
    TypeError
        If the `df` parameter is not a pandas DataFrame. This ensures that the function is applied to the correct data type.

    ValueError
        - If an invalid 'method' option is provided. This error informs users about the acceptable values for the 'method' parameter.
        - If 'buf' parameter is used in the 'info' method. This error is specific to the limitation that 'buf' parameter cannot be used within the 'info' method in `explore_df`.
        - If an invalid 'output' option is provided. This error guides users to choose only from the supported options for 'output'.

    Examples
    --------
    # Create a sample DataFrame to use in the examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> data = {
    ...     'A': np.random.randn(100),
    ...     'B': np.random.rand(100) * 100,
    ...     'C': np.random.randint(1, 100, size=100),
    ...     'D': np.random.choice(['X', 'Y', 'Z'], size=100)
    ... }
    >>> df = pd.DataFrame(data)

    # The power of this function lies in 'all' method, simply:
    >>> explore_df(df, 'all')

    # Alternatively you can save the output to text and even to a file
    >>> summary = explore_df(df, 'all', output='return')

    # Display summary statistics with custom percentiles (thanks to **kwargs filtering!)
    >>> explore_df(df, 'desc', percentiles=[0.05, 0.95], output='print')

    # Display the first 3 rows
    >>> explore_df(df, 'head', n=3, output='print')

    # Detailed DataFrame information
    >>> explore_df(df, 'info', verbose=True, output='print')

    # Count and percentage of missing values
    >>> explore_df(df, 'na', output='print')

    # Comprehensive exploration with custom settings, displaying to console
    >>> explore_df(df, 'all', n=3, percentiles=[0.25, 0.75], output='print')

    # Returning comprehensive exploration results as a string
    >>> result_str = explore_df(df, 'all', n=5, output='return')
    >>> print(result_str)

    # Using 'all' with kwargs that apply to specific methods, printing the results:
    >>> explore_df(df, 'all', n=5, percentiles=[0.1, 0.9], verbose=False, output='print')
    """

    # Error Handling #
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The df parameter must be a pandas DataFrame.")

    valid_methods = ['na', 'desc', 'head', 'info', 'all']
    if method.lower() not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Valid options are: {', '.join(valid_methods)}")

    if 'buf' in kwargs and method.lower() == 'info':
        raise ValueError("'buf' parameter is not supported in the 'info' method within explore_df.")

    if output.lower() not in ['print', 'return']:
        raise ValueError("Invalid output method. Choose 'print' or 'return'.")

    # Main Function Logic #
    result = []

    if method.lower() in ["desc", "all"]:
        desc_kwargs = filter_kwargs('describe', kwargs, valid_kwargs)
        result.append(f"<<______DESCRIBE______>>\n{str(df.describe(**desc_kwargs))}\n")

    if method.lower() in ["head", "all"]:
        head_kwargs = filter_kwargs('head', kwargs, valid_kwargs)
        pd.set_option('display.max_columns', None)
        result.append(f"<<______HEAD______>>\n{str(df.head(**head_kwargs))}\n")
        pd.reset_option('display.max_columns')

    if method.lower() in ["info", "all"]:
        info_kwargs = filter_kwargs('info', kwargs, valid_kwargs)
        buffer = io.StringIO()
        df.info(buf=buffer, **info_kwargs)
        result.append(f"<<______INFO______>>\n{buffer.getvalue()}\n")

    if method.lower() in ["na", "all"]:
        na_count = df.isna().sum()
        na_percent = (df.isna().sum() / df.shape[0]) * 100
        result.append(f"<<______NA_COUNT______>>\n{na_count}\n")
        result.append(f"<<______NA_PERCENT______>>\n{na_percent}\n")

    # Combine all results
    combined_result = "\n".join(result)

    if output.lower() == 'print':
        print(combined_result)
    elif output.lower() == 'return':
        return combined_result
    else:
        raise ValueError("Invalid output method. Choose 'print' or 'return'.")
