import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler
from scipy.stats import boxcox, yeojohnson
from scipy.stats.mstats_basic import winsorize


def transform_num(df: pd.DataFrame, numerical_variables: list, method: str, output_distribution: str = 'normal', n_quantiles: int = 1000, random_state: int = 444, with_centering: bool = True, quantile_range: tuple = (25.0, 75.0), power: float = None, power_map: dict = None, lower_percentile: float = 0.01, upper_percentile: float = 0.99, winsorization_map: dict = None, interaction_pairs: list = None, degree: int = None, degree_map: dict = None, bins: int = None, bin_map: dict = None):
    """
    Applies various numerical data transformations to improve machine learning model performance or data analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the numerical data to transform.
    numerical_variables : list
        A list of column names in `df` that are numerical and will be transformed.
    method : str
        The transformation method to apply. Valid methods include:
        - 'standardize': Mean=0, SD=1. Suitable for algorithms sensitive to variable scales.
        - 'log': Natural logarithm transformation for positively skewed data.
        - 'normalize': Scales data to a [0, 1] range. Useful for models sensitive to variable scales.
        - 'quantile': Transforms data to follow a specified distribution, improving statistical analysis.
        - 'robust': Scales data using the median and quantile range, reducing the influence of outliers.
        - 'boxcox': Normalizes skewed data, requires positive values.
        - 'yeojohnson': Similar to Box-Cox but suitable for both positive and negative values.
        - 'power': Raises numerical variables to specified powers for distribution adjustment.
        - 'winsorization': Caps extreme values to reduce impact of outliers.
        - 'interaction': Creates new features by multiplying pairs of numerical variables.
        - 'polynomial': Generates polynomial features up to a specified degree.
        - 'bin': Groups numerical data into bins or intervals.
    output_distribution : str, optional
        Specifies the output distribution for 'quantile' method ('normal' or 'uniform'). Default is 'normal'.
    n_quantiles : int, optional
        Number of quantiles to use for 'quantile' method. Default is 1000.
    random_state : int, optional
        Random state for 'quantile' method. Default is 444.
    with_centering : bool, optional
        Whether to center data before scaling for 'robust' method. Default is True.
    quantile_range : tuple, optional
        Quantile range used for 'robust' method. Default is (25.0, 75.0).
    power : float, optional
        The power to raise each numerical variable for 'power' method. Default is None.
    power_map : dict, optional
        A dictionary mapping variables to their respective powers for 'power' method. Default is None.
    lower_percentile : float, optional
        Lower percentile for 'winsorization'. Default is 0.01.
    upper_percentile : float, optional
        Upper percentile for 'winsorization'. Default is 0.99.
    winsorization_map : dict, optional
        A dictionary specifying winsorization bounds per variable. Default is None.
    interaction_pairs : list, optional
        List of tuples specifying pairs of variables for creating interaction terms. Default is None.
    degree : int, optional
        The degree for polynomial features in 'polynomial' method. Default is None.
    degree_map : dict, optional
        A dictionary mapping variables to their respective degrees for 'polynomial' method. Default is None.
    bins : int, optional
        The number of equal-width bins to use for 'bin' method. Default is None.
    bin_map : dict, optional
        A dictionary specifying custom binning criteria per variable for 'bin' method. Default is None.

    Returns
    -------
    transformed_df : pd.DataFrame
        The DataFrame with transformed numerical variables.
    transformed_columns : pd.DataFrame
        A DataFrame containing only the transformed columns.

    Examples
    --------
    >>> df = pd.DataFrame({'Feature1': np.random.normal(0, 1, 100), 'Feature2': np.random.exponential(1, 100), 'Feature3': np.random.randint(1, 100, 100)})
    >>> num_cols = ['Feature1', 'Feature2', 'Feature3']

    # Standardize
    >>> standardized_data, standardized_cols = transform_num(df, num_cols, method='standardize')

    # Log transformation
    >>> log_data, log_cols = transform_num(df, num_cols, method='log')

    # Normalize
    >>> normalized_data, normalized_cols = transform_num(df, num_cols, method='normalize')

    # Quantile transformation
    >>> quant_transformed_data, quant_transformed_cols = transform_num(df, num_cols, method='quantile', output_distribution='normal', n_quantiles=1000, random_state=444)

    # Robust scaling
    >>> robust_transformed_df, robust_transformed_columns = transform_num(df, num_cols, method='robust', with_centering=True, quantile_range=(25.0, 75.0))

    # Box-Cox transformation
    >>> boxcox_transformed_df, boxcox_transformed_columns = transform_num(df, num_cols, method='boxcox')

    # Yeo-Johnson transformation
    >>> yeojohnson_transformed_df, yeojohnson_transformed_columns = transform_num(df, num_cols, method='yeojohnson')

    # Power transformation using a uniform power
    >>> power_transformed_df1, power_transformed_columns1 = transform_num(df, num_cols, method='power', power=2)

    # Power transformation using a power map
    >>> power_map = {'Feature1': 2, 'Feature2': 3, 'Feature3': 4}
    >>> power_transformed_df2, power_transformed_columns2 = transform_num(df, num_cols, method='power', power_map=power_map)

    # Winsorization with global thresholds
    >>> wins_transformed_df1, wins_transformed_columns1 = transform_num(df, num_cols, method='winsorization', lower_percentile=0.01, upper_percentile=0.99)

    # Winsorization using a winsorization map
    >>> win_map = {'Feature1': (0.01, 0.99), 'Feature2': (0.05, 0.95), 'Feature3': [0.10, 0.90]}
    >>> wins_transformed_df2, wins_transformed_columns2 = transform_num(df, num_cols, method='winsorization', winsorization_map=win_map)

    # Interaction terms
    >>> interactions = [('Feature1', 'Feature2'), ('Feature2', 'Feature3')]
    >>> inter_transformed_df, inter_columns = transform_num(df, num_cols, method='interaction', interaction_pairs=interactions)

    # Polynomial features with a degree map
    >>> degree_map = {'Feature1': 2, 'Feature2': 3}
    >>> poly_transformed_df, poly_features = transform_num(df, ['Feature1', 'Feature2'], method='polynomial', degree_map=degree_map)

    # Binning with a bin map
    >>> bin_map = {'Feature2': {'bins': 5}, 'Feature3': {'edges': [1, 20, 40, 60, 80, 100]}}
    >>> bin_transformed_df, binned_columns = transform_num(df, ['Feature2', 'Feature3'], method='bin', bin_map=bin_map)
    """

    # explore_num todos #
    # TODO: Implement new method: 'outlier_dbscan' (density-based spatial clustering outlier detection)
    # TODO: Implement new method: 'outlier_isoforest' (isolation forest outlier detection)
    # TODO: Implement new method: 'outlier_lof'(local outlier factor outlier detection)

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
        print(f"✔ Dataframe with only the transformed columns:\n{standardized_columns.head()}\n")
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
        print(f"✔ Dataframe with only the transformed columns:\n{normalized_columns.head()}\n")
        print("☻ HOW TO: Apply this transformation using `transformed_df, normalized_columns = transform_num(your_df, your_numerical_variables, method='normalize')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")

        return transformed_df, normalized_columns

    if method.lower() == 'quantile':
        print(f"< QUANTILE TRANSFORMATION >")
        print(f" This method maps the data to a '{output_distribution}' distribution and n_quantiles = {n_quantiles}. Random state set to {random_state}")
        print(f"  ✔ Transforms skewed or outlier-affected data to follow a standard {'normal' if output_distribution == 'normal' else 'uniform'} distribution, improving statistical analysis and ML model accuracy.")
        print(f"  ✔ Utilizes {n_quantiles} quantiles to finely approximate the empirical distribution, capturing the detailed data structure while balancing computational efficiency.\n")
        print(f"☻ Tip: The choice of 1000 quantiles as a default provides a good compromise between detailed distribution mapping and practical computational demands. Adjust as needed based on dataset size and specificity.\n")

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
        print(f"< ROBUST SCALING TRANSFORMATION >")
        print(f" This method scales your data by removing the median and scaling according to the quantile range.")
        print(f"  ✔ Targets data with outliers by using median and quantiles, reducing the influence of extreme values.")
        print(f"  ✔ Centers and scales data to be robust against outliers, improving model performance on skewed data.")
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
        print(f"< BOX-COX TRANSFORMATION >")
        print(f" This method applies the Box-Cox transformation to numerical variables to normalize their distribution.")
        print(f"  ✔ Transforms skewed data to closely approximate a normal distribution.")
        print(f"  ✔ Automatically finds and applies the optimal transformation parameter (lambda) for each variable.\n")
        print(f"✎ Note: Box-Cox transformation requires all data to be positive. Columns with zero or negative values will be skipped.\n")

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
        print(f"< YEO-JOHNSON TRANSFORMATION >")
        print(f" This method transforms data to closely approximate a normal distribution, applicable to both positive and negative values.")
        print(f"  ✔ Stabilizes variance and normalizes distribution.")
        print(f"  ✔ Suitable for a wide range of data, including zero and negative values.\n")

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
        print(f"< POWER TRANSFORMATION >")
        print(f" This method raises numerical variables to specified powers, allowing for precise data distribution adjustments.")
        print(f"  ✔ Individual powers can be set per variable using a 'power_map' for targeted transformations.")
        print(f"  ✔ Alternatively, a single 'power' value applies uniformly to all specified numerical variables.")
        print(f"  ✔ Facilitates skewness correction and distribution normalization to improve statistical analysis and ML model performance.\n")
        print(f"☻ Tip: A power of 0.5 (square root) often works well for right-skewed data, while a square (power of 2) can help with left-skewed data. Choose the power that best fits your data characteristics.\n")

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
        print(f"☻ HOW TO: Apply this transformation using `transformed_df, power_transformed_columns = transform_num(your_df, your_numerical_variables, method='power', power_map=your_power_map)`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original dataframe: {df.shape}")
        print(f"  ➡ Shape of transformed dataframe: {transformed_df.shape}\n")
        print("* Evaluate the distribution post-transformation to ensure it aligns with your analytical or modeling goals.\n")

        return transformed_df, power_transformed_columns

    if method.lower() == 'winsorization' and (lower_percentile is not None and upper_percentile is not None) or winsorization_map is not None:
        print(f"< WINSORIZATION TRANSFORMATION >")
        print(f" This method caps extreme values in the data to reduce the impact of outliers.")
        print(f"✎ Note: Specify `lower_percentile` and `upper_percentile` for all variables, or use `winsorization_map` for variable-specific thresholds.\n")

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
        print(f"< INTERACTION TERMS TRANSFORMATION >")
        print(f" This method creates new features by multiplying together pairs of numerical variables.")
        print(f"  ✔ Captures the synergistic effects between variables that may impact the target variable.")
        print(f"  ✔ Can unveil complex relationships not observable through individual variables alone.\n")
        print(f"✎ Note: Specify pairs of variables using 'interaction_pairs', which is a list of tuples, where tuples are variable pairs.\n")

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
        print(f"< POLYNOMIAL FEATURES TRANSFORMATION >")
        print(f" This method generates polynomial features up to a specified degree for numerical variables.")
        print(f"  ✔ Captures non-linear relationships between variables and the target.")
        print(f"  ✔ Enhances model performance by adding complexity through feature engineering.")
        print(f"✎ Note: Specify the 'degree' for a global application or 'degree_map' for variable-specific degrees.\n")

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
        print(f"< BINNING TRANSFORMATION >")
        print(f" This method groups numerical data into bins or intervals, simplifying relationships and reducing noise.")
        print(f"  ✔ Users can specify a uniform number of bins for all variables or define custom binning criteria per variable.")
        print(f"✎ Note: Binning can be specified globally with 'bins' or individually with 'bin_map'.\n")

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
                        transformed_df[variable], _ = pd.cut(transformed_df[variable], bins=bin_edges, retbins=True, labels=range(len(bin_edges)-1))
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

# quantile
quant_transformed_data, quant_transformed_cols = transform_num(df, num_cols, method='quantile', output_distribution='normal', n_quantiles=1000, random_state=444)

# robust
robust_transformed_df, robust_transformed_columns = transform_num(df, num_cols, method='robust', with_centering=True, quantile_range=(25.0, 75.0))

# boxcox
boxcox_transformed_df, boxcox_transformed_columns = transform_num(df, num_cols, method='boxcox')

# yeojohnson
yeojohnson_transformed_df, yeojohnson_transformed_columns = transform_num(df, num_cols, method='yeojohnson')

# power

# power usage
power2 = 2
power_transformed_df1, power_transformed_columns1 = transform_num(df, num_cols, method='power', power=power2)

# power_map usage
power_map234 = {
    'Feature1': 2,
    'Feature2': 3,
    'Feature3': 4
}
power_transformed_df2, power_transformed_columns2 = transform_num(df, num_cols, method='power', power_map=power_map234)


# winsorization

# lower_quantile and upper_quantile usage
lower_bound = 0.01
upper_bound = 0.99
wins_transformed_df1, wins_transformed_columns1 = transform_num(df, num_cols, method='winsorization', lower_percentile=lower_bound, upper_percentile=upper_bound)

# winsorization_map usage
win_map = {
    'Feature1': (0.01, 0.99),
    'Feature2': (0.05, 0.95), # tuples work
    'Feature3': [0.10, 0.90] # lists work too
}
wins_transformed_df2, wins_transformed_columns2 = transform_num(df, num_cols, method='power', winsorization_map=win_map)

# interaction
interactions = [
    ('Feature1', 'Feature2'),
    ('Feature2', 'Feature3')
]
inter_transformed_df, inter_columns = transform_num(df, num_cols, method='interaction', interaction_pairs=interactions)

# polynomial
degree_map = {'Feature1': 2, 'Feature2': 3}
poly_transformed_df, poly_features = transform_num(df, ['Feature1', 'Feature2'], method='polynomial', degree_map=degree_map)

# bin
bin_map = {
    'Feature2': {'bins': 5},  # For simplicity, specifying only the number of bins for Feature2
    'Feature3': {'edges': [1, 20, 40, 60, 80, 100]}  # Defining custom bin edges for Feature3
}

# Assuming transform_num is your function to apply binning based on bin_map or a uniform number of bins
bin_transformed_df, binned_columns = transform_num(df, ['Feature2', 'Feature3'], method='bin', bin_map=bin_map)

# This would bin Feature2 into 5 equal-width intervals based on its data range.
# For Feature3, it creates bins based on the specified edges: [1-20), [20-40), [40-60), [60-80), [80-100].
