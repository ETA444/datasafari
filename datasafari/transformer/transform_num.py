from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler
from scipy.stats import boxcox, yeojohnson
from scipy.stats.mstats import winsorize
from datasafari.evaluator.evaluate_dtype import evaluate_dtype


def transform_num(
        df: pd.DataFrame,
        numerical_variables: List[str],
        method: str,
        output_distribution: str = 'normal',
        n_quantiles: int = 1000,
        random_state: int = 444,
        with_centering: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        power: Optional[float] = None,
        power_map: Optional[Dict[str, float]] = None,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
        winsorization_map: Optional[Dict[str, Tuple[float, float]]] = None,
        interaction_pairs: Optional[List[Tuple[str, str]]] = None,
        degree: Optional[int] = None,
        degree_map: Optional[Dict[str, int]] = None,
        bins: Optional[int] = None,
        bin_map: Optional[Dict[str, List[float]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    **Transform numerical variables in a DataFrame through operations like standardization, log-transformation, various scalings, winsorization, interaction term creation and more.**

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the numerical data to transform.

    numerical_variables : list
        A list of column names in `df` that are numerical and will be transformed.

    method : str
        The transformation method to apply.
            - ``'standardize'`` Mean=0, SD=1. Suitable for algorithms sensitive to variable scales.
            - ``'log'`` Natural logarithm transformation for positively skewed data.
            - ``'normalize'`` Scales data to a [0, 1] range. Useful for models sensitive to variable scales.
            - ``'quantile'`` Transforms data to follow a specified distribution, improving statistical analysis.
            - ``'robust'`` Scales data using the median and quantile range, reducing the influence of outliers.
            - ``'boxcox'`` Normalizes skewed data, requires positive values.
            - ``'yeojohnson'`` Similar to Box-Cox but suitable for both positive and negative values.
            - ``'power'`` Raises numerical variables to specified powers for distribution adjustment.
            - ``'winsorization'`` Caps extreme values to reduce impact of outliers.
            - ``'interaction'`` Creates new features by multiplying pairs of numerical variables.
            - ``'polynomial'`` Generates polynomial features up to a specified degree.
            - ``'bin'`` Groups numerical data into bins or intervals.

    output_distribution : str, optional, default: 'normal'
        Specifies the output distribution for 'quantile' method ('normal' or 'uniform').

    n_quantiles : int, optional, default: 1000
        Number of quantiles to use for 'quantile' method.

    random_state : int, optional, default: 444
        Random state for 'quantile' method.

    with_centering : bool, optional, default: True
        Whether to center data before scaling for 'robust' method.

    quantile_range : tuple, optional, default: (25.0, 75.0)
        Quantile range used for 'robust' method.

    power : float, optional, default: None
        The power to raise each numerical variable for 'power' method.

    power_map : dict, optional, default: None
        A dictionary mapping variables to their respective powers for 'power' method.

    lower_percentile : float, optional, default: 0.01
        Lower percentile for 'winsorization'.

    upper_percentile : float, optional, default: 0.99
        Upper percentile for 'winsorization'.

    winsorization_map : dict, optional, default: None
        A dictionary specifying winsorization bounds per variable.

    interaction_pairs : list, optional, default: None
        List of tuples specifying pairs of variables for creating interaction terms.

    degree : int, optional, default: None
        The degree for polynomial features in 'polynomial' method. Default is None.

    degree_map : dict, optional, default: None
        A dictionary mapping variables to their respective degrees for 'polynomial' method.

    bins : int, optional, default: None
        The number of equal-width bins to use for 'bin' method.

    bin_map : dict, optional, default: None
        A dictionary specifying custom binning criteria per variable for 'bin' method.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Original DataFrame with transformed numerical variables.
        - A DataFrame containing only the transformed columns.

    Raises:
    -------
    TypeErrors:
        - If `df` is not a pandas DataFrame.
        - If `numerical_variables` is not a list.
        - If `method` is not a string.
        - If `output_distribution` is provided but not a string.
        - If `n_quantiles` is not an integer.
        - If `random_state` is not an integer.
        - If `with_centering` is not a boolean.
        - If `quantile_range` is not a tuple of two floats.
        - If `power` is provided but not a float.
        - If `power_map`, `winsorization_map`, `degree_map`, or `bin_map` is provided but not a dictionary.
        - If `lower_percentile` or `upper_percentile` is not a float.
        - If `interaction_pairs` is not a list of tuples, or tuples are not of length 2.
        - If `degree` is provided but not an integer.
        - If `bins` is provided but not an integer.

    ValueErrors:
        - If the input DataFrame is empty.
        - If 'numerical_variables' list is empty.
        - If variables provided through 'numerical_variables' are not numerical variables.
        - If any of the specified `numerical_variables` are not found in the DataFrame's columns.
        - If the `method` specified is not one of the valid methods.
        - If `output_distribution` is not 'normal' or 'uniform' for the 'quantile' method.
        - If `n_quantiles` is not a positive integer for the 'quantile' method.
        - If `quantile_range` does not consist of two float values in the range 0 to 1 for the 'robust' method.
        - If `power` is not provided for the 'power' method when required.
        - If `lower_percentile` or `upper_percentile` is not between 0 and 1, or if `lower_percentile` is greater than or equal to `upper_percentile` for the 'winsorization' method.
        - If `degree` is not provided or is not a positive integer for the 'polynomial' method when required.
        - If `bins` is not a positive integer for the 'bin' method when required.
        - If method is 'log', 'boxcox' or 'yeojohnson' and the provided columns have NAs or Infs raise as these statistical methods are not compatible with NAs or Infs.
        - If specified keys in `power_map`, `winsorization_map`, `degree_map`, or `bin_map` do not match any column in the DataFrame.
        - If the `interaction_pairs` specified do not consist of columns that exist in the DataFrame.

    Examples:
    ---------
    Import necessary libraries and generate a DataFrame for examples:

    >>> import datasafari
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Feature1': np.random.normal(0, 1, 100),
    ...     'Feature2': np.random.exponential(1, 100),
    ...     'Feature3': np.random.randint(1, 100, 100)
    ... })
    >>> num_cols = ['Feature1', 'Feature2', 'Feature3']

    Standardize:

    >>> standardized_data, standardized_cols = transform_num(df, num_cols, method='standardize')

    Log transformation:

    >>> log_data, log_cols = transform_num(df, num_cols, method='log')

    Normalize:

    >>> normalized_data, normalized_cols = transform_num(df, num_cols, method='normalize')

    Quantile transformation:

    >>> quant_transformed_data, quant_transformed_cols = transform_num(df, num_cols, method='quantile', output_distribution='normal', n_quantiles=1000, random_state=444)

    Robust scaling:

    >>> robust_transformed_df, robust_transformed_columns = transform_num(df, num_cols, method='robust', with_centering=True, quantile_range=(25.0, 75.0))

    Box-Cox transformation:

    >>> boxcox_transformed_df, boxcox_transformed_columns = transform_num(df, num_cols, method='boxcox')

    Yeo-Johnson transformation:

    >>> yeojohnson_transformed_df, yeojohnson_transformed_columns = transform_num(df, num_cols, method='yeojohnson')

    Power transformation using a uniform power:

    >>> power_transformed_df1, power_transformed_columns1 = transform_num(df, num_cols, method='power', power=2)

    Power transformation using a power map:

    >>> power_map = {'Feature1': 2, 'Feature2': 3, 'Feature3': 4}
    >>> power_transformed_df2, power_transformed_columns2 = transform_num(df, num_cols, method='power', power_map=power_map)

    Winsorization with global thresholds:

    >>> wins_transformed_df1, wins_transformed_columns1 = transform_num(df, num_cols, method='winsorization', lower_percentile=0.01, upper_percentile=0.99)

    Winsorization using a winsorization map:

    >>> win_map = {'Feature1': (0.01, 0.99), 'Feature2': (0.05, 0.95), 'Feature3': [0.10, 0.90]}
    >>> wins_transformed_df2, wins_transformed_columns2 = transform_num(df, num_cols, method='winsorization', winsorization_map=win_map)

    Interaction terms:

    >>> interactions = [('Feature1', 'Feature2'), ('Feature2', 'Feature3')]
    >>> inter_transformed_df, inter_columns = transform_num(df, num_cols, method='interaction', interaction_pairs=interactions)

    Polynomial features with a degree map:

    >>> degree_map = {'Feature1': 2, 'Feature2': 3}
    >>> poly_transformed_df, poly_features = transform_num(df, ['Feature1', 'Feature2'], method='polynomial', degree_map=degree_map)

    Binning with a bin map:

    >>> bin_map = {'Feature2': {'bins': 5}, 'Feature3': {'edges': [1, 20, 40, 60, 80, 100]}}
    >>> bin_transformed_df, binned_columns = transform_num(df, ['Feature2', 'Feature3'], method='bin', bin_map=bin_map)
    """

    # Error-handling #

    # TypeErrors
    # Check if 'df' is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("transform_num(): The 'df' parameter must be a pandas DataFrame.")

    # Check if 'numerical_variables' is a list
    if not isinstance(numerical_variables, list):
        raise TypeError("transform_num(): The 'numerical_variables' parameter must be a list of column names.")
    else:
        if not all(isinstance(var, str) for var in numerical_variables):
            raise TypeError("transform_num(): All elements in the 'numerical_variables' list must be strings representing column names.")

    # Check if 'method' is a string
    if not isinstance(method, str):
        raise TypeError("transform_num(): The 'method' parameter must be a string.")

    # Check if 'output_distribution' is a string
    if not isinstance(output_distribution, str):
        raise TypeError("transform_num(): The 'output_distribution' parameter must be a string.")

    # Check if 'n_quantiles' is an integer
    if not isinstance(n_quantiles, int):
        raise TypeError("transform_num(): The 'n_quantiles' parameter must be an integer.")

    # Check if 'random_state' is an integer
    if not isinstance(random_state, int):
        raise TypeError("transform_num(): The 'random_state' parameter must be an integer.")

    # Check if 'with_centering' is a boolean
    if not isinstance(with_centering, bool):
        raise TypeError("transform_num(): The 'with_centering' parameter must be a boolean.")

    # Check if 'quantile_range' is a tuple and contains two floats
    if not (isinstance(quantile_range, tuple) and len(quantile_range) == 2 and all(isinstance(num, float) for num in quantile_range)):
        raise TypeError("transform_num(): The 'quantile_range' parameter must be a tuple containing two float values.")

    # Check if 'power' is None, int, or float
    if power is not None and not isinstance(power, (int, float)):
        raise TypeError("transform_num(): The 'power' parameter must be a float, integer, or None.")

    # Check if 'power_map' is None or a dictionary
    if power_map is not None and not isinstance(power_map, dict):
        raise TypeError("transform_num(): The 'power_map' parameter must be a dictionary or None.")

    # Check if 'lower_percentile' and 'upper_percentile' are floats
    if not isinstance(lower_percentile, float) or not isinstance(upper_percentile, float):
        raise TypeError("transform_num(): The 'lower_percentile' and 'upper_percentile' parameters must be floats.")

    # Check if 'winsorization_map' is None or a dictionary
    if winsorization_map is not None and not isinstance(winsorization_map, dict):
        raise TypeError("transform_num(): The 'winsorization_map' parameter must be a dictionary or None.")

    # Check if 'interaction_pairs' is None or a list of tuples
    if interaction_pairs is not None:
        if not (isinstance(interaction_pairs, list) and all(isinstance(pair, tuple) and len(pair) == 2 for pair in interaction_pairs)):
            raise TypeError("transform_num(): The 'interaction_pairs' parameter must be a list of tuples or None.")

    # Check if 'degree' is None or an integer
    if degree is not None and not isinstance(degree, int):
        raise TypeError("transform_num(): The 'degree' parameter must be an integer or None.")

    # Check if 'degree_map' is None or a dictionary
    if degree_map is not None and not isinstance(degree_map, dict):
        raise TypeError("transform_num(): The 'degree_map' parameter must be a dictionary or None.")

    # Check if 'bins' is None or an integer
    if bins is not None and not isinstance(bins, int):
        raise TypeError("transform_num(): The 'bins' parameter must be an integer or None.")

    # Check if 'bin_map' is None or a dictionary
    if bin_map is not None and not isinstance(bin_map, dict):
        raise TypeError("transform_num(): The 'bin_map' parameter must be a dictionary or None.")

    # ValueErrors
    # Check if df is empty
    if df.empty:
        raise ValueError("explore_num(): The input DataFrame is empty.")

    # Check if list has any members
    if len(numerical_variables) == 0:
        raise ValueError("transform_num(): The 'numerical_variables' list must contain at least one column name.")

    # Check if variables are numerical
    numerical_types = evaluate_dtype(df, numerical_variables, output='list_n')
    if not all(numerical_types):
        raise ValueError("transform_num(): The 'numerical_variables' list must contain only names of numerical variables.")

    # Check if specified variables exist in the DataFrame
    missing_vars = [var for var in numerical_variables if var not in df.columns]
    if missing_vars:
        raise ValueError(f"transform_num(): The following numerical variables were not found in the DataFrame: {', '.join(missing_vars)}")

    # Check if method is valid
    valid_methods = ['standardize', 'log', 'normalize', 'quantile', 'robust', 'boxcox', 'yeojohnson', 'power', 'winsorization', 'interaction', 'polynomial', 'bin']
    if method.lower() not in valid_methods:
        raise ValueError(f"transform_num(): Invalid method '{method}'. Valid options are: {', '.join(valid_methods)}")

    # For 'quantile' method specific checks
    if method.lower() == 'quantile':
        if output_distribution not in ['normal', 'uniform']:
            raise ValueError("transform_num(): Invalid 'output_distribution' for 'quantile' method. Choose 'normal' or 'uniform'.")
        if not isinstance(n_quantiles, int) or n_quantiles <= 0:
            raise ValueError("transform_num(): The 'n_quantiles' must be a positive integer.")
        if not isinstance(random_state, int):
            raise ValueError("transform_num(): The 'random_state' must be an integer.")

    # For 'robust' method specific checks
    if method.lower() == 'robust':
        if not isinstance(with_centering, bool):
            raise ValueError("transform_num(): The 'with_centering' parameter must be a boolean (True or False).")
        if not (isinstance(quantile_range, tuple) and len(quantile_range) == 2 and all(isinstance(num, float) for num in quantile_range)):
            raise ValueError("transform_num(): The 'quantile_range' must be a tuple of two float values.")

    # For 'power' method specific checks
    if method.lower() == 'power':
        if power is not None and not isinstance(power, (float, int)):
            raise ValueError("transform_num(): The 'power' parameter must be a float, integer, or None.")
        if power_map is not None and not isinstance(power_map, dict):
            raise ValueError("transform_num(): The 'power_map' must be a dictionary mapping variables to powers or None.")

    # For 'winsorization' method specific checks
    if method.lower() == 'winsorization':
        if not (isinstance(lower_percentile, float) and 0 <= lower_percentile < 1):
            raise ValueError("transform_num(): The 'lower_percentile' must be a float between 0 and 1.")
        if not (isinstance(upper_percentile, float) and 0 < upper_percentile <= 1):
            raise ValueError("transform_num(): The 'upper_percentile' must be a float between 0 and 1.")
        if lower_percentile >= upper_percentile:
            raise ValueError("transform_num(): The 'lower_percentile' must be less than 'upper_percentile'.")

    # For 'polynomial' method specific checks
    if method.lower() == 'polynomial':
        if degree is not None and not (isinstance(degree, int) and degree > 0):
            raise ValueError("transform_num(): The 'degree' must be a positive integer or None.")
        if degree_map is not None and not isinstance(degree_map, dict):
            raise ValueError("transform_num(): The 'degree_map' must be a dictionary mapping variables to degrees or None.")

    # For 'bin' method specific checks
    if method.lower() == 'bin':
        if bins is not None and not (isinstance(bins, int) and bins > 0):
            raise ValueError("transform_num(): The 'bins' must be a positive integer or None.")
        if bin_map is not None and not isinstance(bin_map, dict):
            raise ValueError("transform_num(): The 'bin_map' must be a dictionary specifying binning criteria or None.")

    # For 'interaction' method specific checks
    if method.lower() == 'interaction':
        if interaction_pairs is not None:
            if not (isinstance(interaction_pairs, list) and all(isinstance(pair, tuple) and len(pair) == 2 for pair in interaction_pairs)):
                raise ValueError("transform_num(): The 'interaction_pairs' must be a list of tuples specifying pairs of variables or None.")
            missing_pairs = [pair for pair in interaction_pairs if pair[0] not in df.columns or pair[1] not in df.columns]
            if missing_pairs:
                raise ValueError(f"transform_num(): The following variable pairs in 'interaction_pairs' were not found in the DataFrame: {missing_pairs}")

    # Check for NaN or infinite values in the DataFrame for methods that cannot handle them
    if method.lower() in ['log', 'boxcox', 'yeojohnson']:
        if df[numerical_variables].isnull().values.any() or np.isinf(df[numerical_variables].values).any():
            raise ValueError(f"transform_num(): The 'numerical_variables' contain NaN or infinite values, which are not compatible with the '{method}' method.")

    # Additional checks for mapping dictionaries to ensure keys exist in the DataFrame
    if power_map or winsorization_map or degree_map or bin_map:
        for mapping, map_name in zip([power_map, winsorization_map, degree_map, bin_map], ['power_map', 'winsorization_map', 'degree_map', 'bin_map']):
            if mapping:
                invalid_keys = [key for key in mapping.keys() if key not in df.columns]
                if invalid_keys:
                    raise ValueError(f"transform_num(): The following keys in '{map_name}' were not found in the DataFrame columns: {', '.join(invalid_keys)}")

    # Main Function #
    if method == 'standardize':
        print("< STANDARDIZING DATA >")
        print(" This method centers the data around mean 0 with a standard deviation of 1, enhancing model performance and stability.")
        print("  ✔ Standardizes each numerical variable to have mean=0 and variance=1.")
        print("  ✔ Essential preprocessing step for many machine learning algorithms.\n")
        print("✎ Note: Standardization is applied only to the specified numerical variables.\n")

        # initialize essential objects
        transformed_df = df.copy()
        scaler = StandardScaler()

        # scale the data
        transformed_df[numerical_variables] = scaler.fit_transform(df[numerical_variables])

        # isolate transformed columns to give as part of output
        standardized_columns = transformed_df[numerical_variables]

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the transformed columns:\n{standardized_columns.head()}\n")
        print("☻ HOW TO - Apply this transformation using `transformed_df, standardized_columns = transform_num(your_df, your_numerical_variables, method='standardize')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")

        return transformed_df, standardized_columns

    if method.lower() == 'log':
        print("< LOG TRANSFORMATION >")
        print(" This method applies a natural logarithm transformation to positively skewed data.")
        print("  ✔ Helps to stabilize variance and make the data more normally distributed.")
        print("  ✔ Particularly useful for data with a heavy right tail (positively skewed).\n")
        print("✎ Note: Log transformation is applied only to specified numerical variables. Zero or negative values in the data can cause issues and will skip those columns.\n")

        # initialize essential objects
        transformed_df = df.copy()
        log_transformed_columns = pd.DataFrame()
        skipped_columns = []

        # do the operation only after checking provided column for zero or negative values
        for variable in numerical_variables:
            if (transformed_df[variable] <= 0).any():
                print(f"⚠️ Warning: '{variable}' contains zero or negative values and was skipped to avoid log(0) and negative log issues.\n")
                print("☻ Note: The log transformation requires strictly positive values. For data with zero or negative values, consider using the 'yeojohnson' method as an alternative.")
                skipped_columns.append(variable)
            else:
                transformed_column = np.log(transformed_df[variable])
                transformed_df[variable] = transformed_column
                log_transformed_columns = pd.concat([log_transformed_columns, transformed_column], axis=1)
                print(f"✔ '{variable}' has been log-transformed.\n")

        # inform user if a column was skipped
        if skipped_columns:
            print(f"✘ Skipped columns due to non-positive values: {skipped_columns}\n")

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the transformed columns:\n{log_transformed_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, log_transformed_columns = transform_num(your_df, your_numerical_variables, method='log')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")
        print("* Consider examining the distribution of your data post-transformation.\n")

        return transformed_df, log_transformed_columns

    if method.lower() == 'normalize':
        print("< NORMALIZATION TRANSFORMATION >")
        print(" This method scales numerical variables to a [0, 1] range, making them suitable for models sensitive to variable scales.")
        print("  ✔ Adjusts each feature to a [0, 1] scale based on its minimum and maximum values.")
        print("  ✔ Enhances model performance by ensuring numerical variables are on a similar scale.")
        print("✎ Note: Ensure data is clean and outliers are handled for optimal results.\n")
        print("☻ Tip: Use `explore_num()` for data inspection and outlier detection before applying normalization.\n")

        # initialize essentials
        transformed_df = df.copy()
        min_max_scaler = MinMaxScaler()

        # apply minmax scaling
        transformed_df[numerical_variables] = min_max_scaler.fit_transform(df[numerical_variables])

        # isolate transformed columns to provide as part of output
        normalized_columns = transformed_df[numerical_variables]

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the transformed columns:\n{normalized_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, normalized_columns = transform_num(your_df, your_numerical_variables, method='normalize')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")

        return transformed_df, normalized_columns

    if method.lower() == 'quantile':
        print("< QUANTILE TRANSFORMATION >")
        print(f" This method maps the data to a '{output_distribution}' distribution and n_quantiles = {n_quantiles}. Random state set to {random_state}")
        print(f"  ✔ Transforms skewed or outlier-affected data to follow a standard {'normal' if output_distribution == 'normal' else 'uniform'} distribution, improving statistical analysis and ML model accuracy.")
        print(f"  ✔ Utilizes {n_quantiles} quantiles to finely approximate the empirical distribution, capturing the detailed data structure while balancing computational efficiency.\n")
        print("☻ Tip: The choice of 1000 quantiles as a default provides a good compromise between detailed distribution mapping and practical computational demands. Adjust as needed based on dataset size and specificity.\n")

        # initialize the DataFrame to work with
        transformed_df = df.copy()

        # define and apply Quantile Transformer
        quantile_transformer = QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles, random_state=random_state)
        transformed_df[numerical_variables] = quantile_transformer.fit_transform(df[numerical_variables])

        # isolate transformed columns to give as part of output
        quantile_transformed_columns = transformed_df[numerical_variables]

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the transformed columns:\n{quantile_transformed_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, quantile_transformed_columns = transform_num(your_df, your_numerical_variables, method='quantile', output_distribution='normal', n_quantiles=1000, random_state=444)`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")
        print("* After transformation, evaluate your data's distribution and consider its impact on your analysis or modeling approach.\n")

        return transformed_df, quantile_transformed_columns

    if method.lower() == 'robust':
        print("< ROBUST SCALING TRANSFORMATION >")
        print(" This method scales your data by removing the median and scaling according to the quantile range.")
        print("  ✔ Targets data with outliers by using median and quantiles, reducing the influence of extreme values.")
        print("  ✔ Centers and scales data to be robust against outliers, improving model performance on skewed data.")
        print(f"✎ Note: With centering is {'enabled' if with_centering else 'disabled'}. Adjust `with_centering` as needed (provide bool).\n")
        print(f"☻ Tip: The quantile range is set to {quantile_range}. You can adjust it based on your data's distribution.\n")

        # initialize essentials
        transformed_df = df.copy()
        scaler = RobustScaler(with_centering=with_centering, quantile_range=quantile_range)

        # apply the defined robust scaler
        transformed_df[numerical_variables] = scaler.fit_transform(transformed_df[numerical_variables])

        # Isolate transformed columns to give as part of output
        robust_scaled_columns = transformed_df[numerical_variables]

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the transformed columns:\n{robust_scaled_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, robust_scaled_columns = transform_num(your_df, your_numerical_variables, method='robust', with_centering=True, quantile_range=(25.0, 75.0))`.\n")

        # Sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")
        print("* After transformation, evaluate the robustness of your data against outliers and consider the effect on your analysis or modeling.\n")

        return transformed_df, robust_scaled_columns

    if method.lower() == 'boxcox':
        print("< BOX-COX TRANSFORMATION >")
        print(" This method applies the Box-Cox transformation to numerical variables to normalize their distribution.")
        print("  ✔ Transforms skewed data to closely approximate a normal distribution.")
        print("  ✔ Automatically finds and applies the optimal transformation parameter (lambda) for each variable.\n")
        print("✎ Note: Box-Cox transformation requires all data to be positive. Columns with zero or negative values will be skipped.\n")

        # initialize essential objects
        transformed_df = df.copy()
        boxcox_transformed_columns = pd.DataFrame()
        skipped_columns = []

        # check column values and apply boxcox only on appropriate data
        for variable in numerical_variables:
            if (transformed_df[variable] <= 0).any():
                print(f"⚠️ Warning: '{variable}' contains zero or negative values and was skipped to avoid issues with Box-Cox transformation.\n")
                print("☻ Tip: The Box-Cox transformation requires strictly positive values. For data including zero or negative values, the 'yeojohnson' method provides a flexible alternative.")
                skipped_columns.append(variable)
            else:
                transformed_column, _ = boxcox(transformed_df[variable])
                transformed_df[variable] = transformed_column
                boxcox_transformed_columns = pd.concat([boxcox_transformed_columns, pd.DataFrame(transformed_column, columns=[variable])], axis=1)
                print(f"✔ '{variable}' has been Box-Cox transformed.\n")

        # inform user of any columns that were skipped
        if skipped_columns:
            print(f"✘ Skipped columns due to non-positive values: {', '.join(skipped_columns)}\n")

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the Box-Cox transformed columns:\n{boxcox_transformed_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, boxcox_transformed_columns = transform_num(your_df, your_numerical_variables, method='boxcox')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")
        print("* After transformation, evaluate your data's distribution and consider its impact on your analysis or modeling approach.\n")

        return transformed_df, boxcox_transformed_columns

    if method.lower() == 'yeojohnson':
        print("< YEO-JOHNSON TRANSFORMATION >")
        print(" This method transforms data to closely approximate a normal distribution, applicable to both positive and negative values.")
        print("  ✔ Stabilizes variance and normalizes distribution.")
        print("  ✔ Suitable for a wide range of data, including zero and negative values.\n")

        # initialize the DataFrame to work with
        transformed_df = df.copy()
        yeojohnson_transformed_columns = pd.DataFrame()

        # Apply Yeo-Johnson transformation
        for variable in numerical_variables:
            transformed_column, lambda_value = yeojohnson(transformed_df[variable])
            transformed_df[variable] = transformed_column
            yeojohnson_transformed_columns[variable] = transformed_column
            print(f"✔ '{variable}' has been transformed using Yeo-Johnson. Lambda value: {lambda_value:.4f}\n")

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the Yeo-Johnson transformed columns:\n{yeojohnson_transformed_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, yeojohnson_transformed_columns = transform_num(your_df, your_numerical_variables, method='yeojohnson')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")
        print("* Review the transformed data and consider its implications for your analysis or modeling strategy.\n")

        return transformed_df, yeojohnson_transformed_columns

    if method.lower() == 'power' and (power is not None or power_map is not None):
        print("< POWER TRANSFORMATION >")
        print(" This method raises numerical variables to specified powers, allowing for precise data distribution adjustments.")
        print("  ✔ Individual powers can be set per variable using a 'power_map' for targeted transformations.")
        print("  ✔ Alternatively, a single 'power' value applies uniformly to all specified numerical variables.")
        print("  ✔ Facilitates skewness correction and distribution normalization to improve statistical analysis and ML model performance.\n")
        print("☻ Tip: A power of 0.5 (square root) often works well for right-skewed data, while a square (power of 2) can help with left-skewed data. Choose the power that best fits your data characteristics.\n")

        # initialize essential objects
        transformed_df = df.copy()
        power_transformed_columns = pd.DataFrame()

        # determine transformation approach
        if power_map is not None:
            for variable, pwr in power_map.items():
                if variable in numerical_variables:
                    transformed_column = np.power(transformed_df[variable], pwr)
                    transformed_df[variable] = transformed_column
                    power_transformed_columns = pd.concat([power_transformed_columns, transformed_column], axis=1)
                    print(f"✔ '{variable}' has been transformed with a power of {pwr}.\n")
        else:
            for variable in numerical_variables:
                transformed_column = np.power(transformed_df[variable], power)
                transformed_df[variable] = transformed_column
                power_transformed_columns = pd.concat([power_transformed_columns, transformed_column], axis=1)
                print(f"✔ '{variable}' uniformly transformed with a power of {power}.\n")

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the power transformed columns:\n{power_transformed_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, power_transformed_columns = transform_num(your_df, your_numerical_variables, method='power', power_map=your_power_map)`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")
        print("* Evaluate the distribution post-transformation to ensure it aligns with your analytical or modeling goals.\n")

        return transformed_df, power_transformed_columns

    if method.lower() == 'winsorization' and (lower_percentile is not None and upper_percentile is not None) or winsorization_map is not None:
        print("< WINSORIZATION TRANSFORMATION >")
        print(" This method caps extreme values in the data to reduce the impact of outliers.")
        print("✎ Note: Specify `lower_percentile` and `upper_percentile` for all variables, or use `winsorization_map` for variable-specific thresholds.\n")

        # initialize objects
        transformed_df = df.copy()
        winsorized_columns = pd.DataFrame()

        # determine transformation approach
        if winsorization_map is not None:
            for variable, bounds in winsorization_map.items():
                if variable in numerical_variables:
                    transformed_column = winsorize(transformed_df[variable], limits=(bounds[0], 1 - bounds[1]))
                    transformed_df[variable] = transformed_column
                    winsorized_columns[variable] = transformed_column
                    print(f"✔ '{variable}' has been winsorized with bounds {bounds}.\n")
                else:
                    print(f"⚠️ Warning: '{variable}' specified in `winsorization_map` was not found in `numerical_variables` and has been skipped.\n")
        else:
            for variable in numerical_variables:
                transformed_column = winsorize(transformed_df[variable], limits=(lower_percentile, 1 - upper_percentile))
                transformed_df[variable] = transformed_column
                winsorized_columns[variable] = transformed_column
                print(f"✔ '{variable}' has been winsorized with lower percentile = {lower_percentile*100}% and upper percentile = {upper_percentile*100}%.\n")

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the power transformed columns:\n{winsorized_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, winsorized_columns = transform_num(your_df, your_numerical_variables, method='winsorization', lower_percentile=0.01, upper_percentile=0.99)` for global thresholds, or provide `winsorization_map` for variable-specific thresholds.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")
        print("* After transformation, review the distribution of your numerical variables to ensure outliers have been appropriately capped.\n")

        return transformed_df, winsorized_columns

    if method.lower() == 'interaction':
        print("< INTERACTION TERMS TRANSFORMATION >")
        print(" This method creates new features by multiplying together pairs of numerical variables.")
        print("  ✔ Captures the synergistic effects between variables that may impact the target variable.")
        print("  ✔ Can unveil complex relationships not observable through individual variables alone.\n")
        print("✎ Note: Specify pairs of variables using 'interaction_pairs', which is a list of tuples, where tuples are variable pairs.\n")

        # initialize essential objects
        transformed_df = df.copy()
        interaction_columns = pd.DataFrame()

        # check if interaction_pairs is provided
        if not interaction_pairs:
            print("⚠️ No 'interaction_pairs' provided. Please specify pairs of variables for interaction terms. Where the object is a list of tuples, and each tuple is a pair of variables.")
            return transformed_df, pd.DataFrame()

        # create interaction terms
        for pair in interaction_pairs:
            new_column_name = f"interaction_{pair[0]}_x_{pair[1]}"
            interaction_columns[new_column_name] = transformed_df[pair[0]] * transformed_df[pair[1]]
            print(f"✔ Created interaction term '{new_column_name}' based on variables '{pair[0]}' and '{pair[1]}'.\n")

        # Add interaction columns to the transformed DataFrame
        transformed_df = pd.concat([transformed_df, interaction_columns], axis=1)

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the interaction columns:\n{interaction_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, interaction_columns = transform_num(your_df, your_numerical_variables, method='interaction', interaction_pairs=[('var1', 'var2'), ('var3', 'var4')])`.\n")

        # Sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Original dataframe shape: {df.shape}")
        print(f"  ➡ Transformed dataframe shape: {transformed_df.shape}\n")

        return transformed_df, interaction_columns

    if method.lower() == 'polynomial' and (degree is not None or degree_map is not None):
        print("< POLYNOMIAL FEATURES TRANSFORMATION >")
        print(" This method generates polynomial features up to a specified degree for numerical variables.")
        print("  ✔ Captures non-linear relationships between variables and the target.")
        print("  ✔ Enhances model performance by adding complexity through feature engineering.")
        print("✎ Note: Specify the 'degree' for a global application or 'degree_map' for variable-specific degrees.\n")

        # initialize essential objects
        transformed_df = df.copy()
        poly_features = pd.DataFrame(index=df.index)

        # function to apply either global degree or degree from degree_map
        def apply_degree(variable, d):
            for power in range(2, d + 1):  # Start from 2 as degree 1 is the original variable
                new_column_name = f"{variable}_degree_{power}"
                poly_features[new_column_name] = transformed_df[variable] ** power
                print(f"✔ Created polynomial feature '{new_column_name}' from variable '{variable}' to the power of {power}.\n")

        # check if degree_map is provided and apply
        if degree_map:
            for variable, var_degree in degree_map.items():
                if variable in numerical_variables:
                    apply_degree(variable, var_degree)
                else:
                    print(f"⚠️ Variable '{variable}' specified in `degree_map` was not found in `numerical_variables` and has been skipped.\n")
        # if degree_map is not provided, use global degree for all variables
        else:
            for variable in numerical_variables:
                apply_degree(variable, degree)

        # Add polynomial features to the transformed DataFrame
        transformed_df = pd.concat([transformed_df, poly_features], axis=1)

        print(f"✔ New transformed dataframe with polynomial features:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the polynomial features:\n{poly_features.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, poly_features = transform_num(your_df, your_numerical_variables, method='polynomial', degree=3)` or by specifying a `degree_map`.\n")

        # Sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Original dataframe shape: {df.shape}")
        print(f"  ➡ Transformed dataframe shape: {transformed_df.shape}\n")
        print("* After applying polynomial features, evaluate the model's performance and watch out for overfitting, especially when using high degrees.\n")

        return transformed_df, poly_features

    if method.lower() == 'bin' and (bins is not None or bin_map is not None):
        print("< BINNING TRANSFORMATION >")
        print(" This method groups numerical data into bins or intervals, simplifying relationships and reducing noise.")
        print("  ✔ Users can specify a uniform number of bins for all variables or define custom binning criteria per variable.")
        print("✎ Note: Binning can be specified globally with 'bins' or individually with 'bin_map'.\n")

        transformed_df = df.copy()
        binned_columns = pd.DataFrame()

        if bins:
            # Apply a uniform number of bins across all variables
            for variable in numerical_variables:
                transformed_df[variable], bin_edges = pd.cut(transformed_df[variable], bins, retbins=True, labels=range(bins))
                binned_columns = pd.concat([binned_columns, transformed_df[variable]], axis=1)
                print(f"✔ '{variable}' has been binned into {bins} intervals.\n")
        elif bin_map:
            # Apply custom binning based on the provided map
            for variable, specs in bin_map.items():
                if variable in numerical_variables:
                    n_bins = specs.get('bins')
                    bin_edges = specs.get('edges', None)
                    if bin_edges:
                        transformed_df[variable], _ = pd.cut(transformed_df[variable], bins=bin_edges, retbins=True, labels=range(len(bin_edges) - 1))
                    else:
                        transformed_df[variable], _ = pd.cut(transformed_df[variable], bins=n_bins, retbins=True, labels=range(n_bins))
                    binned_columns = pd.concat([binned_columns, transformed_df[variable]], axis=1)
                    print(f"✔ '{variable}' has been custom binned based on provided specifications.\n")

        print(f"✔ New transformed dataframe with binned variables:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the binned columns:\n{binned_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, binned_columns = transform_num(your_df, your_numerical_variables, method='bin', bins=3)` or use bin_map.\n")

        # Sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")
        print("* Review the binned data to ensure it aligns with your analysis or modeling strategy.\n")

        return transformed_df, binned_columns
