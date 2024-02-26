import pandas as pd
from scipy.stats import shapiro, skew, kurtosis, anderson


# main function: explore_num
def explore_num(df: pd.DataFrame, numerical_variables: list, method: str = 'all', output: str = 'print'):
    """
    Analyze numerical variables in a DataFrame for distribution characteristics, outliers based on Z-score and IQR methods, and normality tests.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the numerical data to analyze.
    numerical_variables : list
        A list of strings representing the column names in `df` to be analyzed.
    method : str, optional
        Specifies the analysis method to apply. Options include:
        - 'distribution_analysis' for distribution characteristics and normality tests.
        - 'outliers_zscore' for outlier detection using the Z-score method.
        - 'outliers_iqr' for outlier detection using the Interquartile Range method.
        - 'all' to perform all analyses. Default is 'all'.
    output : str, optional
        Determines the output format. Options include:
        - 'print' to print the analysis results to the console.
        - 'return' to return the analysis results as a DataFrame or dictionaries. Default is 'print'.

    Returns
    -------
    Depending on the method chosen:
    - For 'distribution_analysis', returns a DataFrame with each row representing a statistic (min, max, mean, etc.) and columns for each variable.
    - For 'outliers_zscore' and 'outliers_iqr', returns two objects: a dictionary mapping variables to their outlier values, and a DataFrame of rows in `df` considered outliers.
    - If 'output' is set to 'return' and 'method' is 'all', returns a textual summary of all analyses.

    Raises
    ------
    ValueError
        If an invalid 'method' or 'output' option is provided.

    Notes
    -----
    - The function enhances interpretability by providing tips and conclusions based on the statistical tests conducted.
    - For normality tests:
        * Shapiro-Wilk Test: H0 (null hypothesis) suggests that the data is normally distributed. A p-value > 0.05 supports H0.
        * Anderson-Darling Test: Compares the test statistic to critical values to assess normality.
    - Skewness indicates the symmetry of the distribution. Kurtosis indicates the tailedness of the distribution.

    Examples
    --------
    # Example 1: Exploring distribution characteristics and printing to console
    >>> df = pd.read_csv('path/to/dataset.csv')
    >>> cols = ['variable1', 'variable2']
    >>> explore_num(df, cols, method='distribution_analysis', output='print')

    # Example 2: Detecting outliers using the Z-score method and returning the results
    >>> outliers_z_dict, outliers_z_df = explore_num(df, ['variable3', 'variable4'], method='outliers_zscore', output='return')
    >>> print(outliers_z_dict)
    >>> print(outliers_z_df.head())

    # Example 3: Identifying outliers with the IQR method and printing the summary
    >>> explore_num(df, ['variable5', 'variable6'], method='outliers_iqr', output='print')

    # Example 4: Performing all analyses on specified variables and printing summaries
    >>> explore_num(df, ['variable1', 'variable2', 'variable3'], method='all', output='print')

    # Example 5: Using 'return' with 'all' method to get a comprehensive textual summary
    >>> analysis_summary = explore_num(df, ['variable7', 'variable8'], method='all', output='return')
    >>> print(analysis_summary)

    # Example 6: Exploring distribution for a single variable and capturing the DataFrame for further analysis
    >>> distribution_df = explore_num(df, ['variable9'], method='distribution_analysis', output='return')
    >>> print(distribution_df)

    # Example 7: Comprehensive analysis for multiple variables, focusing on outlier detection and distribution, and returning data for IQR outliers
    >>> outliers_iqr_dict, outliers_iqr_df = explore_num(df, cols, method='outliers_iqr', output='return')
    >>> distribution_analysis_df = explore_num(df, cols, method='distribution_analysis', output='return')
    >>> print(outliers_iqr_df.head())
    >>> print(distribution_analysis_df)

    """

    # initialize variables #
    # (1) each method appends to: result
    result = []
    # (2) method 'outliers_zscore' only, returns these to the user
    outliers_z_dict = {}
    outliers_z_df = pd.DataFrame()
    # (3) method 'outliers_iqr' only, returns these to the user
    outliers_iqr_dict = {}
    outliers_iqr_df = pd.DataFrame()
    # (4) method 'distribution_analysis' only, returns this df to the user
    distribution_df = pd.DataFrame(columns=numerical_variables)
    # (5) method 'correlation_analysis' only, returns df to the user
    correlation_dfs = []

    # TODO: New method: 'assumptions' - Add general model assumption tests
    # # # TODO: Migrate Normality tests to assumptions.
    # TODO: New method: 'outliers_mahalanobis' - Add additional more robust outlier check: mahalanobis
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
            result.append(f"\n<<______CORRELATIONS ({correlation_method.upper()})______>>\n")
            result.append(f"Overview of {correlation_method.title()} Correlation Coefficients*\n")
            result.append(correlation_df.to_string())

        # save dataframes for extended functionality (method='correlation_analysis')
        correlation_dfs = [pearson_df, spearman_df, kendall_df]

    if method.lower() in ['distribution_analysis', 'all']:

        # appends #
        # (1) title of method section
        result.append(f"<<______DISTRIBUTION ANALYSIS______>>\n")
        # (2) subtitle
        result.append(f"✎ Overview of Results*\n")

        # define #
        # define dist stats for dictionary
        stats_functions = ['min', 'max', 'mean', 'median', 'mode', 'variance', 'std_dev', 'skewness', 'kurtosis', 'shapiro_p', 'anderson_stat']
        # initialize dictionary to be used in the creation of distribution_df
        stats_dict = {stat: [] for stat in stats_functions}

        # main operation: descriptive stats, skewness, kurtosis, normality testing
        for variable_name in numerical_variables:

            # modify data for this analysis: no NAs
            data = df[variable_name].dropna()

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
            skewness_tip = f"   ☻ Tip: Symmetric if ~0, left-skewed if <0, right-skewed if >0"
            kurt_tip = f"   ☻ Tip: Mesokurtic if ~0, Platykurtic if <0, Leptokurtic if >0)"
            # shapiro-wilk interpretation and conclusion based on p
            shapiro_tip = f"  • H0: Data is normally distributed.\n  • H1: Data is not normally distributed."
            shapiro_conclusion = f"✘ Conclusion: ['{variable_name}'] is likely not normally distributed." if shapiro_p < 0.05 else "\n✔ Conclusion: ['{variable_name}'] is likely normally distributed."
            # anderson-darling
            anderson_tip = f"   ☻ Tip: Compare the statistic to critical values. Data is likely not normally distributed if the statistic > critical value."

            # construct console output
            result.append(f"< Distribution Analysis Summary for: ['{variable_name}'] >\n")
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
            stats_dict['mode'].append(mode[0] if len(mode) != 0 else pd.NA) # handle special case of multiple modes
            stats_dict['variance'].append(variance)
            stats_dict['std_dev'].append(std_dev)
            stats_dict['skewness'].append(skewness)
            stats_dict['kurtosis'].append(kurt)
            stats_dict['shapiro_p'].append(shapiro_p)
            stats_dict['anderson_stat'].append(anderson_stat.statistic)

        # construct df from stats_dict: distribution_df
        for stat, values in stats_dict.items():
            distribution_df = pd.concat(
                [distribution_df, pd.DataFrame({numerical_variables[i]: [values[i]] for i in range(len(values))}, index=[stat])]
            )

        # transpose df for easier readability (wide): index is variable name, column is statistic
        distribution_df = distribution_df.T
        distribution_df.columns = stats_functions
        distribution_df.index.name = 'Variable/Statistic'

        # appends (continued) #
        # method='distribution_analysis' info if method is all
        if method.lower() == 'all':
            result.append(f"✎ * NOTE: If method='distribution_analysis', aside from the overview above, the function RETURNS:")
            result.append(f"■ 1 - Dataframe: where index are your variables, columns are all the calculated statistic (wide format for readability)")
            result.append(f"☻ HOW TO: df = explore_num(yourdf, yourlist, method='distribution_analysis')")

    if method.lower() in ['outliers_iqr', 'all']:

        # appends #
        # (1) title of method section
        result.append(f"<<______OUTLIERS - IQR METHOD______>>\n")
        # (2) suitability tip
        result.append(f"Tip: The IQR method is robust against extreme values, ideal for identifying outliers\nin skewed distributions by focusing on the data's middle 50%.\n")
        # (3) subtitle
        result.append(f"✎ Overview of Results*\n")

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
                row_indices = f"➡ Location of outliers in your df (indices): -\n"
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
            result.append(f"✎ * NOTE: If method='outliers_iqr', aside from the overview above, the function RETURNS:")
            result.append(f"■ 1 - Dictionary: key=variable name, value=list of outlier values for that row")
            result.append(f"■ 2 - Dataframe: Rows from the original df that were classified as outliers. (preserved index)")
            result.append(f"☻ HOW TO: dict, df = explore_num(yourdf, yourlist, method='outliers_iqr')")

    if method.lower() in ['outliers_zscore', 'all']:

        # definitions #
        # (1) z-score threshold for outlier detection
        threshold_z = 3

        # appends #
        # (1) title of method section
        result.append(f"<<______OUTLIERS - Z-SCORE METHOD______>>\n")
        # (2) suitability tip
        result.append(f"Tip: The Z-Score method excels at identifying outliers in data with a distribution\nclose to normal, highlighting values far from the mean.\n")
        # (3) subtitle
        result.append(f"✎ Overview of Results*\n")

        # main operation: z-score calculation per variable, and outlier classification.
        for variable_name in numerical_variables:

            # zscore column name
            z_col = variable_name + '_zscore'

            # calculate z-score for col
            df[z_col] = (
                    (df[variable_name] - df[variable_name].mean())
                    / df[variable_name].std()
            )

            # outlier classification
            outlier_rows = df[
                df[z_col].abs() > threshold_z
            ]

            # save results: dictionary and df (objects)
            outliers_z_dict[variable_name] = outlier_rows[variable_name].tolist()
            outliers_z_df = pd.concat([outliers_z_df, outlier_rows], ignore_index=False)

            # clean the original df
            df.drop(z_col, axis=1, inplace=True)

            # conditional output string format and stats calculations
            title = f"< Results for ['{variable_name}'] >\n"
            result.append(title)
            outlier_count = len(outliers_z_dict[variable_name])
            if outlier_count == 0:
                stats = f"➡ Number of outliers: {outlier_count}\n➡ Min: -\n➡ Max: -\n➡ Mean: -"
                row_indices = f"➡ Location of outliers in your df (indices): -\n"
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
            result.append(f"✎ * NOTE: If method='outliers_zscore', aside from the overview above, the function RETURNS:")
            result.append(f"■ 1 - Dictionary: key=variable name, value=list of outlier values for that row")
            result.append(f"■ 2 - Dataframe: Rows from the original df that were classified as outliers. (preserved index)")
            result.append(f"☻ HOW TO: dict, df = explore_num(yourdf, yourlist, method='outliers_zscore')")

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

        if method.lower() == 'distribution_analysis':
            return distribution_df

        if method.lower() == 'correlation_analysis':
            return correlation_dfs

    elif output.lower() == 'return':
        # normal functionality of output: return
        if method.lower() == 'all':
            return combined_result

        # extended functionality of output: return (method-specific)
        if method.lower() == 'outliers_zscore':
            return outliers_z_dict, outliers_z_df

        if method.lower() == 'outliers_iqr':
            return outliers_iqr_dict, outliers_iqr_df

        if method.lower() == 'distribution_analysis':
            return distribution_df

        if method.lower() == 'correlation_analysis':
            return correlation_dfs

    else:
        raise ValueError("Invalid output method. Choose 'print' or 'return'.")


# smoke tests
pengu = pd.read_csv('./datasets/penguins.csv')
cols = [
    'culmen_length_mm', 'culmen_depth_mm',
    'flipper_length_mm', 'body_mass_g'
]

pearson_df, spearman_df, kendall_df = explore_num(pengu, cols, method='correlation_analysis')
# outlier_dict, outlier_df = explore_num(pengu, cols, method='outliers_zscore')
