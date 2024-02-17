# imports
import pandas as pd


# module function
def explore_df(df: pd.DataFrame, method: object):
    """
    Function to run describe, head, or info on df.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to explore.
    method : {'desc', 'head', 'info', 'all'}
        Specify the method to use.
        - 'desc': Display summary statistics using describe().
        - 'head': Display the first few rows using head().
        - 'info': Display concise information about the DataFrame using info().
        - 'na': Display counts of NAs per column and percentage of NAs per column.
        - 'all': Display all information from above options.

    Returns
    -------
    None
    """
    if method.lower() == "desc":
        print(df.describe())
    elif method.lower() == "head":
        pd.set_option('display.max_columns', None)
        print(df.head())
        pd.reset_option('display.max_columns')
    elif method.lower() == "info":
        print(df.info())
    elif method.lower() == "na":
        print(f"\n\n<<______NA_COUNT______>>")
        print(df.isna().sum())
        print(f"\n\n<<______NA_PERCENT______>>")
        print((df.isna().sum() / df.shape[0]) * 100)
    elif method.lower() == "all":
        print("<<______HEAD______>>")
        pd.set_option('display.max_columns', None)
        print(df.head())
        pd.reset_option('display.max_columns')
        print(f"\n\n<<______DESCRIBE______>>")
        print(df.describe())
        print(f"\n\n<<______INFO______>>")
        print(df.info())
        print(f"\n\n<<______NA_COUNT______>>")
        print(df.isna().sum())
        print(f"\n\n<<______NA_PERCENT______>>")
        print((df.isna().sum() / df.shape[0]) * 100)
    else:
        print("Methods: 'desc', 'head', 'info' or 'all'")


# smoke test
# tdf = pd.read_csv('./data/soil_measures.csv')
# explore_df(tdf, 'all')
