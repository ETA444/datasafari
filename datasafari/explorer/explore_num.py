# DataSafari - DataSafari simplifies complex data science tasks into straightforward, powerful one-liners.
# Copyright (C) 2024 George Dreemer.
#
# Read more about DataSafari's LICENSE here: https://datasafari.dev/docs/other/lic-gpl3
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details, specifically under
# version 3 of the License. No later versions are applicable.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see: https://github.com/ETA444/datasafari/blob/main/LICENSE

from typing import List, Optional, Tuple, Dict
import numpy as np
from numpy.linalg import inv
import pandas as pd
from scipy.stats import (
    shapiro, skew, kurtosis, anderson,  # used in 'distribution_analysis'
    chi2  # used in 'outliers_mahalanobis'
)
from datasafari.utils.calculators import calculate_mahalanobis, calculate_vif
from datasafari.evaluator.evaluate_dtype import evaluate_dtype


# explore_num todos 2.0 #
# TODO: Implement new method: 'outlier_dbscan' (density-based spatial clustering outlier detection)
# TODO: Implement new method: 'outlier_isoforest' (isolation forest outlier detection)
# TODO: Implement new method: 'outlier_lof'(local outlier factor outlier detection)
# TODO: Overhaul explore_num output methodology to be more like transformer module one


# main function: explore_num
def explore_num(
        df: pd.DataFrame,
        numerical_variables: List[str],
        method: str = 'all',
        output: str = 'print',
        threshold_z: int = 3
) -> Optional[Tuple[Dict, pd.DataFrame]]:
    """
    **Explore numerical variables in a DataFrame and gain insights on distribution characteristics, outlier detection using multiple methods (Z-score, IQR, Mahalanobis), normality tests, skewness, kurtosis, correlation analysis, and multicollinearity detection.**

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the numerical data to analyze.

    numerical_variables : list
        A list of strings representing the column names in `df` to be analyzed.

    method : str, optional, default: 'all'
        Specifies the analysis method to apply.
            - ``'correlation_analysis'`` for analyzing the correlation between numerical variables.
            - ``'distribution_analysis'`` for distribution characteristics, including skewness and kurtosis, and normality tests (Shapiro-Wilk, Anderson-Darling).
            - ``'outliers_zscore'`` for outlier detection using the Z-score method.
            - ``'outliers_iqr'`` for outlier detection using the Interquartile Range method.
            - ``'outliers_mahalanobis'`` for outlier detection using the Mahalanobis distance.
            - ``'multicollinearity'`` for detecting multicollinearity among the numerical variables.
            - ``'all'`` to perform all available analyses.

    output : str, optional, default: 'print'
        Determines the output format.
            - ``'print'`` to print the analysis results to the console.
            - ``'return'`` to return the analysis results as a DataFrame or dictionaries, depending on the analysis type.

    threshold_z : int, optional, default; 3
        Used in method ``'outliers_zscore'``, users can define their preferred z-score threshold, if the default value does not fit their needs.

    Returns:
    --------
    Tuple[Dict, pd.DataFrame] or None.
        - ``Tuple[Dict, pd.DataFrame]`` For 'correlation_analysis', returns a DataFrame showing the correlation coefficients between variables if output is 'return'.
        - ``Tuple[Dict, pd.DataFrame]`` For 'distribution_analysis', returns a DataFrame with distribution statistics if output is 'return'.
        - ``Tuple[Dict, pd.DataFrame]`` For outlier detection methods ('outliers_zscore', 'outliers_iqr', 'outliers_mahalanobis'), returns a dictionary mapping variables to their outlier values and a DataFrame of rows considered outliers if output is 'return'.
        - ``Tuple[Dict, pd.DataFrame]`` For 'multicollinearity', returns a DataFrame or a Series indicating the presence of multicollinearity, such as VIF scores, if output is 'return'.
        - ``Tuple[Dict, pd.DataFrame]`` If output='return' and method='all', returns a comprehensive summary of all analyses as text or a combination of DataFrames and dictionaries.
        - ``None`` If output='print' and method='all', returns nothing, but prints results to console.

    Raises:
    -------
    TypeErrors:
        - If `df` is not a pandas DataFrame.
        - If `numerical_variables` is not a list of strings.
        - If `method` is not a string.
        - If `output` is not a string.
        - If `threshold_z` is not a float or an int.

    ValueErrors:
        - If the `df` is empty.
        - If `method` is not one of the specified valid methods.
        - If `output` is not 'print' or 'return'.
        - If 'numerical_variables' list is empty.
        - If variables provided through 'numerical_variables' are not numerical variables.
        - If any specified variables in `numerical_variables` are not found in the DataFrame's columns.

    Examples:
    ---------
    Generating a sample DataFrame to demonstrate the functionality:

    >>> import datasafari
    >>> import numpy as np
    >>> import pandas as pd
    >>> data = {
    ...    'Feature1': np.random.normal(loc=0, scale=1, size=100),
    ...    'Feature2': np.random.exponential(scale=2, size=100),
    ...    'Feature3': np.random.randint(low=1, high=100, size=100)
    ... }
    >>> df = pd.DataFrame(data)

    The full potential of ``explore_num()`` is unlocked by simply providing a dataframe and the numerical columns to explore:

    >>> explore_num(df, ['Feature1', 'Feature2', 'Feature3'])

    Performing correlation analysis and printing the results:

    >>> explore_num(df, ['Feature1', 'Feature2', 'Feature3'], method='correlation_analysis', output='print')

    Conducting distribution analysis and returning the results:

    >>> distribution_results = explore_num(df, ['Feature1', 'Feature2', 'Feature3'], method='distribution_analysis', output='return')
    >>> print(distribution_results)

    Detecting outliers using the IQR method and printing the results:

    >>> explore_num(df, ['Feature1', 'Feature2', 'Feature3'], method='outliers_iqr', output='print')

    Detecting outliers using the Z-score method with a custom threshold:

    >>> explore_num(df, ['Feature1', 'Feature2', 'Feature3'], method='outliers_zscore', threshold_z=2, output='print')

    Identifying multicollinearity and printing VIF scores:

    >>> explore_num(df, ['Feature1', 'Feature2', 'Feature3'], method='multicollinearity', output='print')

    Applying all analyses and printing comprehensive results:

    >>> explore_num(df, ['Feature1', 'Feature2', 'Feature3'], method='all', output='print')

    Notes
    -----
    - Enhances interpretability by providing insights and conclusions based on the statistical tests and analyses conducted.
    - Normality tests assess whether data distribution departs from a normal distribution, which is crucial for certain statistical analyses.
    - Correlation analysis examines the strength and direction of relationships between numerical variables.
    - Multicollinearity detection is essential for regression analysis, as high multicollinearity can invalidate the model.
    """

    # Error Handling #
    # TypeErrors
    if not isinstance(df, pd.DataFrame):
        raise TypeError("explore_num(): The df parameter must be a pandas DataFrame.")

    if not isinstance(numerical_variables, list):
        raise TypeError("explore_num(): The categorical_variables parameter must be a list of variable names.\n Example: var_list = ['var1', 'var2', 'var3']")
    else:
        if not all(isinstance(var, str) for var in numerical_variables):
            raise TypeError("explore_num(): All items in the numerical_variables list must be strings representing column names.")

    if not isinstance(method, str):
        raise TypeError("explore_num(): The method parameter must be a string.\n Example: method = 'all'")

    if not isinstance(output, str):
        raise TypeError("explore_num(): The output parameter must be a string. \n Example: output = 'return'")

    if not isinstance(threshold_z, (float, int)):
        raise TypeError("explore_num(): The value of threshold_z must be a float or int.\nExample: threshold_z = 3")

    # ValueErrors
    # Check if df is empty
    if df.empty:
        raise ValueError("explore_num(): The input DataFrame is empty.")

    # Check if method is valid
    valid_methods = ['correlation_analysis', 'distribution_analysis', 'outliers_zscore', 'outliers_iqr', 'outliers_mahalanobis', 'multicollinearity', 'all']
    if method.lower() not in valid_methods:
        raise ValueError(f"explore_num(): Invalid method '{method}'. Valid options are: {', '.join(valid_methods)}")

    # Check if output is valid
    if output.lower() not in ['print', 'return']:
        raise ValueError("explore_num(): Invalid output method. Choose 'print' or 'return'.")

    # Check if list has any members
    if len(numerical_variables) == 0:
        raise ValueError("explore_num(): The 'numerical_variables' list must contain at least one column name.")

    # Check if variables are numerical
    numerical_types = evaluate_dtype(df, numerical_variables, output='list_n')
    if not all(numerical_types):
        raise ValueError("explore_num(): The 'numerical_variables' list must contain only names of numerical variables.")

    # Check if specified variables exist in the DataFrame
    missing_vars = [var for var in numerical_variables if var not in df.columns]
    if missing_vars:
        raise ValueError(f"explore_num(): The following variables were not found in the DataFrame: {', '.join(missing_vars)}")

    # Main Function #
    # initialize variables #
    # (1) each method appends to: result
    result = []
    # (2) method 'outliers_zscore' only, returns these to the user
    outliers_z_dict = {}
    outliers_z_df = pd.DataFrame()
    # (3) method 'outliers_iqr' only, returns these to the user
    outliers_iqr_dict = {}
    outliers_iqr_df = pd.DataFrame()
    # (4) method 'outliers_mahalanobis', returns these to the user
    outliers_mahalanobis_df = pd.DataFrame()
    # (5) method 'distribution_analysis' only, returns this df to the user
    distribution_df = pd.DataFrame(columns=numerical_variables)
    # (6) method 'correlation_analysis' only, returns df to the user
    correlation_dfs = []

    if method.lower() in ['correlation_analysis', 'all']:

        # calculate correlations per method
        pearson_df = df[numerical_variables].corr(method='pearson')
        spearman_df = df[numerical_variables].corr(method='spearman')
        kendall_df = df[numerical_variables].corr(method='kendall')

        # define dictionary for for-loop
        correlation_analysis = {
            'pearson': pearson_df,
            'spearman': spearman_df,
            'kendall': kendall_df
        }

        # construct console output per method
        for correlation_method, correlation_df in correlation_analysis.items():
            result.append(f"\n<______CORRELATIONS ({correlation_method.upper()})______>\n")
            result.append(f"Overview of {correlation_method.title()} Correlation Coefficients*\n")
            result.append(correlation_df.to_string())

        # save dataframes for extended functionality (method='correlation_analysis')
        correlation_dfs = [pearson_df, spearman_df, kendall_df]

    if method.lower() in ['distribution_analysis', 'all']:

        # appends #
        # (1) title of method section
        result.append("\n<______DISTRIBUTION ANALYSIS______>\n")
        # (2) subtitle
        result.append("✎ Overview of Results*")

        # define #
        # define dist stats for dictionary
        stats_functions = ['min', 'max', 'mean', 'median', 'mode', 'variance', 'std_dev', 'skewness', 'kurtosis', 'shapiro_p', 'anderson_stat']
        # initialize dictionary to be used in the creation of distribution_df
        stats_dict = {stat: [] for stat in stats_functions}

        # main operation: descriptive stats, skewness, kurtosis, normality testing
        for variable_name in numerical_variables:

            # modify data for this analysis: no NAs
            data = df[variable_name].copy()
            data = data.dropna()

            # calculate descriptive stats
            var_min, var_max = data.min(), data.max()
            mean = data.mean()
            median = data.median()
            mode = data.mode().tolist()
            variance = data.var()
            std_dev = data.std()

            # calculate skewness, kurtosis and tests for normality
            skewness = skew(data)
            kurt = kurtosis(data)
            shapiro_stat, shapiro_p = shapiro(data)
            anderson_stat = anderson(data)

            # interpretation tips #
            # skewness and kurtosis
            skewness_tip = "   ☻ Tip: Symmetric if ~0, left-skewed if <0, right-skewed if >0"
            kurt_tip = "   ☻ Tip: Mesokurtic if ~0, Platykurtic if <0, Leptokurtic if >0)"
            # shapiro-wilk interpretation and conclusion based on p
            shapiro_tip = "  • H0: Data is normally distributed.\n  • H1: Data is not normally distributed."
            shapiro_conclusion = f"✘ Conclusion: ['{variable_name}'] is likely not normally distributed." if shapiro_p < 0.05 else "\n✔ Conclusion: ['{variable_name}'] is likely normally distributed."
            # anderson-darling
            anderson_tip = "   ☻ Tip: Compare the statistic to critical values. Data is likely not normally distributed if the statistic > critical value."

            # construct console output
            result.append(f"\n< Distribution Analysis Summary for: ['{variable_name}'] >\n")
            result.append(f"➡ Min: {var_min:.2f}\n➡ Max: {var_max:.2f}\n➡ Mean: {mean:.2f}\n➡ Median: {median:.2f}\n➡ Mode(s): {mode}")
            result.append(f"➡ Variance: {variance:.2f}\n➡ Standard Deviation: {std_dev:.2f}")
            result.append(f"\n➡ Skewness: {skewness:.2f}\n{skewness_tip}\n\n➡ Kurtosis: {kurt:.2f}\n{kurt_tip}")

            # this output is migrated to 'assumptions' method if user-defined method is 'all'
            if method.lower() == 'distribution_analysis':
                result.append(f"\n★ Shapiro-Wilk Test for Normality:\n{shapiro_tip}\n     ➡ p-value = {shapiro_p:.4f}\n     {shapiro_conclusion}\n")
                result.append(f"\n★ Anderson-Darling Test for Normality:\n   ➡ statistic = {anderson_stat.statistic:.4f}\n   ➡ significance levels = {anderson_stat.significance_level}\n   ➡ critical values = {anderson_stat.critical_values}\n{anderson_tip}\n")

            # save calculation results to stats_dict
            stats_dict['min'].append(var_min)
            stats_dict['max'].append(var_max)
            stats_dict['mean'].append(mean)
            stats_dict['median'].append(median)
            stats_dict['mode'].append(mode[0] if len(mode) != 0 else pd.NA)  # handle special case of multiple modes
            stats_dict['variance'].append(variance)
            stats_dict['std_dev'].append(std_dev)
            stats_dict['skewness'].append(skewness)
            stats_dict['kurtosis'].append(kurt)
            stats_dict['shapiro_p'].append(shapiro_p)
            stats_dict['anderson_stat'].append(anderson_stat.statistic)

        # construct df from stats_dict: distribution_df
        distribution_dfs = []
        for stat, values in stats_dict.items():
            temp_df = pd.DataFrame({stat: values}, index=numerical_variables)
            distribution_dfs.append(temp_df)

        distribution_df = pd.concat(distribution_dfs, axis=1)
        distribution_df.columns = stats_functions  # ensure the columns are named correctly
        distribution_df.index.name = 'Variable/Statistic'

        # appends (continued) #
        # method='distribution_analysis' info if method is all
        if method.lower() == 'all':
            result.append("\n✎ * NOTE: If method='distribution_analysis', aside from the overview above, the function RETURNS:")
            result.append("■ 1 - Dataframe: where index are your variables, columns are all the calculated statistic (wide format for readability)")
            result.append("☻ HOW TO: df = explore_num(yourdf, yourlist, method='distribution_analysis')")

    if method.lower() in ['outliers_iqr', 'all']:

        # appends #
        # (1) title of method section
        result.append("\n<______OUTLIERS - IQR METHOD______>\n")
        # (2) suitability tip
        result.append("☻ Tip: The IQR method is robust against extreme values, ideal for identifying outliers\nin skewed distributions by focusing on the data's middle 50%.\n")
        # (3) subtitle
        result.append("✎ Overview of Results*\n")

        # main operation: quantile definitions, iqr and outlier classification
        for variable_name in numerical_variables:

            # calculate quantile 1, quantile 3 and inter quartile range for respective column
            quantile1 = df[variable_name].quantile(0.25)
            quantile3 = df[variable_name].quantile(0.75)
            iqr = quantile3 - quantile1

            # determine lower and upper bounds
            lower_bound = quantile1 - 1.5 * iqr
            upper_bound = quantile3 + 1.5 * iqr

            # outlier classification
            outlier_rows = df[
                (df[variable_name] < lower_bound) | (df[variable_name] > upper_bound)
            ]

            # save results: dictionary and df (objects)
            outliers_iqr_dict[variable_name] = outlier_rows[variable_name].tolist()
            outliers_iqr_df = pd.concat([outliers_iqr_df, outlier_rows], ignore_index=False)

            # conditional output string format and stats calculations
            title = f"< Results for ['{variable_name}'] >\n"
            result.append(title)
            outlier_count = len(outliers_iqr_dict[variable_name])
            if outlier_count == 0:
                stats = f"➡ Number of outliers: {outlier_count}\n➡ Min: -\n➡ Max: -\n➡ Mean: -"
                row_indices = "➡ Location of outliers in your df (indices): -\n"
                result.append(stats)
                result.append(row_indices)
            else:
                outlier_min = min(outliers_iqr_dict[variable_name])
                outlier_max = max(outliers_iqr_dict[variable_name])
                outlier_mean = sum(outliers_iqr_dict[variable_name]) / len(outliers_iqr_dict[variable_name])
                stats = f"➡ Number of outliers: {outlier_count}\n➡ Min: {outlier_min}\n➡ Max: {outlier_max}\n➡ Mean: {outlier_mean}"
                row_indices = f"➡ Location of outliers in your df (indices):\n{outlier_rows.index.tolist()}\n"
                result.append(stats)
                result.append(row_indices)

        # appends (continued) #
        # (6-9) method='outliers_iqr' info
        if method.lower() == 'all':
            result.append("\n✎ * NOTE: If method='outliers_iqr', aside from the overview above, the function RETURNS:")
            result.append("■ 1 - Dictionary: key=variable name, value=list of outlier values for that row")
            result.append("■ 2 - Dataframe: Rows from the original df that were classified as outliers. (preserved index)")
            result.append("☻ HOW TO: dict, df = explore_num(yourdf, yourlist, method='outliers_iqr')")

    if method.lower() in ['outliers_zscore', 'all']:

        # definitions #
        data = df.copy()

        # appends #
        # (1) title of method section
        result.append("\n<______OUTLIERS - Z-SCORE METHOD______>\n")
        # (2) suitability tip
        result.append("☻ Tip: The Z-Score method excels at identifying outliers in data with a distribution\nclose to normal, highlighting values far from the mean.\n")
        # (3) subtitle
        result.append("✎ Overview of Results*\n")

        # main operation: z-score calculation per variable, and outlier classification.
        for variable_name in numerical_variables:

            # zscore column name
            z_col = variable_name + '_zscore'

            # calculate z-score for col
            data[z_col] = (
                (data[variable_name] - data[variable_name].mean()) / data[variable_name].std()
            )

            # outlier classification
            outlier_rows = data[
                data[z_col].abs() > threshold_z
            ]

            # save results: dictionary and df (objects)
            outliers_z_dict[variable_name] = outlier_rows[variable_name].tolist()
            outliers_z_df = pd.concat([outliers_z_df, outlier_rows], ignore_index=False)

            # conditional output string format and stats calculations
            title = f"< Results for ['{variable_name}'] >\n"
            result.append(title)
            outlier_count = len(outliers_z_dict[variable_name])
            if outlier_count == 0:
                stats = f"➡ Number of outliers: {outlier_count}\n➡ Min: -\n➡ Max: -\n➡ Mean: -"
                row_indices = "➡ Location of outliers in your df (indices): -\n"
                result.append(stats)
                result.append(row_indices)
            else:
                outlier_min = min(outliers_z_dict[variable_name])
                outlier_max = max(outliers_z_dict[variable_name])
                outlier_mean = sum(outliers_z_dict[variable_name]) / len(outliers_z_dict[variable_name])
                stats = f"➡ Number of outliers: {outlier_count}\n➡ Min: {outlier_min}\n➡ Max: {outlier_max}\n➡ Mean: {outlier_mean}"
                row_indices = f"➡ Location of outliers in your df (indices):\n{outlier_rows.index.tolist()}\n"
                result.append(stats)
                result.append(row_indices)

        # appends (continued) #
        # (6-9) method='outliers_zscore' info
        if method.lower() == 'all':
            result.append("\n✎ * NOTE: If method='outliers_zscore', aside from the overview above, the function RETURNS:")
            result.append("■ 1 - Dictionary: key=variable name, value=list of outlier values for that row")
            result.append("■ 2 - Dataframe: Rows from the original df that were classified as outliers. (preserved index)")
            result.append("☻ HOW TO: dict, df = explore_num(yourdf, yourlist, method='outliers_zscore')")

    if method.lower() in ['outliers_mahalanobis', 'all']:
        # definitions #
        # use non-na df: data
        data = df.copy()
        data = data[numerical_variables].dropna()

        try:
            # calculate the mean and inverse of the covariance matrix
            mean_vector = data.mean().values
            inv_cov_matrix = inv(np.cov(data, rowvar=False))

            # apply the utility function to calculate Mahalanobis distance for each observation
            data['mahalanobis'] = data.apply(lambda row: calculate_mahalanobis(row.values, mean_vector, inv_cov_matrix), axis=1)

            # determine outliers based on the chi-square distribution
            p_value_threshold = 0.05
            critical_value = chi2.ppf((1 - p_value_threshold), df=len(numerical_variables))

            # classify outliers based on mahalanobis distance relative to critical value
            outliers_mahalanobis_df = data[data['mahalanobis'] > critical_value]

            # clean up df
            data.drop(columns=['mahalanobis'], inplace=True)

            # construct console output
            result.append("\n<______OUTLIERS - MAHALANOBIS METHOD*______>\n")
            result.append(f"Identified outliers based on Mahalanobis distance exceeding the critical value ({critical_value:.2f}) from the chi-square distribution (p-val < {p_value_threshold}.\n")
            result.append(outliers_mahalanobis_df.to_string())

            # appends (continued) #
            # (6-9) method='outliers_mahalanobis' info
            if method.lower() == 'all':
                result.append("\n✎ * NOTE: If method='outliers_mahalanobis', aside from the overview above, the function RETURNS:")
                result.append("■ 1 - Dataframe: Rows from the original df that were classified as outliers. (preserved index)")
                result.append("☻ HOW TO: df = explore_num(yourdf, yourlist, method='outliers_mahalanobis')")

        except np.linalg.LinAlgError as error:
            result.append(f"Error calculating Mahalanobis distance: {error}")

    if method.lower() in ['multicollinearity', 'all']:

        # use non-na df: data
        data = df.copy()
        data = data[numerical_variables].dropna()

        vifs = calculate_vif(data, numerical_variables)
        result.append("\n<______MULTICOLLINEARITY CHECK - VIF______>\n")
        result.append(f"Variance Inflation Factors:\n{vifs.to_string()}\n")
        result.append("☻ Tip: VIF > 10 indicates potential multicollinearity concerns.")

    # Combine all results
    combined_result = "\n".join(result)

    if output.lower() == 'print':
        # print by default/unconditionally
        print(combined_result)

        # extended functionality of output: return (method-specific)
        if method.lower() == 'outliers_zscore':
            return outliers_z_dict, outliers_z_df

        if method.lower() == 'outliers_iqr':
            return outliers_iqr_dict, outliers_iqr_df

        if method.lower() == 'outliers_mahalanobis':
            return outliers_mahalanobis_df

        if method.lower() == 'distribution_analysis':
            return distribution_df

        if method.lower() == 'correlation_analysis':
            return correlation_dfs

    elif output.lower() == 'return':
        # normal functionality of output: return
        if method.lower() in ['all', 'multicollinearity']:
            return combined_result

        # extended functionality of output: return (method-specific)
        if method.lower() == 'outliers_zscore':
            return outliers_z_dict, outliers_z_df

        if method.lower() == 'outliers_iqr':
            return outliers_iqr_dict, outliers_iqr_df

        if method.lower() == 'outliers_mahalanobis':
            return outliers_mahalanobis_df

        if method.lower() == 'distribution_analysis':
            return distribution_df

        if method.lower() == 'correlation_analysis':
            return correlation_dfs

    else:
        raise ValueError("Invalid output method. Choose 'print' or 'return'.")
