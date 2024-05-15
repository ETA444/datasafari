from typing import Optional
import pandas as pd
import io
from datasafari.utils.filters import filter_kwargs


# dictionary for filter_kwargs: define which kwargs are valid for which method
valid_kwargs = {
    'head': ['n'],
    'describe': ['percentiles', 'include', 'exclude'],
    'info': ['verbose', 'max_cols', 'memory_usage', 'show_counts']
}


# main function: explore_df
def explore_df(
        df: pd.DataFrame,
        method: str = 'all',
        output: str = 'print',
        **kwargs
) -> Optional[str]:
    """
    Gain a quick birds-eye view of a dataframe by checking summary statistics, NAs, data types and more. The function combines the most common data exploration functions in one convenient output in your console.


    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be explored.

    method : str, optional, default: 'all'
        Specifies the method to apply on the DataFrame.

        - ``'na'``: Displays counts of NAs per column and percentage of NAs.
        - ``'desc'``: Shows summary statistics using the `describe` method.
        - ``'head'``: Outputs the first few rows using `head`.
        - ``'info'``: Provides concise information about the DataFrame using `info`.
        - ``'all'``: Executes all the above methods sequentially.

    output : str, optional, default: 'print'
        Determines the output of the exploration results.

        - ``'print'``: Prints the results to the console.
        - ``'return'``: Returns the results as a string.

    **kwargs : dict
        Additional arguments for pandas methods (e.g., 'percentiles' for 'desc').
        You can specify arguments applicable when 'method' is set to 'all', which will be
        appropriately directed to each pandas method used. Note that the 'buf' parameter
        in the 'info' method is disabled and cannot be used.

    Returns
    -------
    str or None
        Returns a string if `output` is 'return'. Otherwise, prints to the console and returns None.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
        If `method` is not a string.
        If `output` is not a string.

    ValueError
        If `df` is empty, indicating no data to evaluate.
        If `method` is not one of the valid options: 'na', 'desc', 'head', 'info', or 'all'.
        If `output` is not 'print' or 'return'.
        If 'buf' parameter is used in the 'info' method.

    Examples
    --------
    Create a sample DataFrame to use in the examples::

        >>> import datasafari
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = {
            'A': np.random.randn(100),
            'B': np.random.rand(100) * 100,
            'C': np.random.randint(1, 100, size=100),
            'D': np.random.choice(['X', 'Y', 'Z'], size=100)
        }
        >>> df = pd.DataFrame(data)

    The full potential of ``explore_df()`` is unlocked by simply providing a dataframe::

        >>> explore_df(df)

    Alternatively, save the output to a string::

        >>> summary = explore_df(df, 'all', output='return')

    Display summary statistics with custom percentiles::

        >>> explore_df(df, 'desc', percentiles=[0.05, 0.95], output='print')

    Show the first 3 rows of the DataFrame::

        >>> explore_df(df, 'head', n=3, output='print')

    Provide detailed DataFrame information::

        >>> explore_df(df, 'info', verbose=True, output='print')

    Calculate and display the count and percentage of missing values::

        >>> explore_df(df, 'na', output='print')

    Execute a comprehensive exploration with custom settings::

        >>> explore_df(df, 'all', n=3, percentiles=[0.25, 0.75], output='print')

    Return comprehensive exploration results as a string::

        >>> result_str = explore_df(df, 'all', n=5, output='return')
        >>> print(result_str)

    Use 'all' with kwargs applicable to specific methods, print the results::

        >>> explore_df(df, 'all', n=5, percentiles=[0.1, 0.9], verbose=False, output='print')
    """



    # Error Handling #
    # TypeErrors for each parameter
    if not isinstance(df, pd.DataFrame):
        raise TypeError("explore_df(): The df parameter must be a pandas DataFrame.")

    if not isinstance(method, str):
        raise TypeError(f"explore_df(): The method parameter must be a string.\nExample: method = 'all'")

    if not isinstance(output, str):
        raise TypeError(f"explore_df(): The output parameter must be a string.\nExample: output = 'print'")

    # ValueErrors
    # Check if df is empty
    if df.empty:
        raise ValueError("explore_df(): The input DataFrame is empty.")

    # Check for correct method
    valid_methods = ['na', 'desc', 'head', 'info', 'all']
    if method.lower() not in valid_methods:
        raise ValueError(f"explore_df(): Invalid method '{method}'. Valid options are: {', '.join(valid_methods)}.")

    # Check for 'output'
    if output.lower() not in ['print', 'return']:
        raise ValueError("explore_df(): Invalid output method. Choose 'print' or 'return'.")

    # Check for unsupported 'info' kwargs
    if 'buf' in kwargs and method.lower() == 'info':
        raise ValueError("explore_df(): 'buf' parameter is not supported in the 'info' method within explore_df.")

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
