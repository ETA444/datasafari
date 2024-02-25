import pandas as pd
from scipy.stats import shapiro, skew, kurtosis, anderson


# main function: explore_num
def explore_num(df: pd.DataFrame, numerical_variables: list, method: str = 'all', output: str = 'print'):
    """..."""

    # initialize variables #
    # (1) each method appends to: result
    result = []
    # (2) method 'outliers_zscore' only, returns these to the user
    outliers_z_dict = {}
    outliers_z_df = pd.DataFrame()
    # (3) method 'outliers_iqr' only, returns these to the user
    outliers_iqr_dict = {}
    outliers_iqr_df = pd.DataFrame()

    if method.lower() in ['distribution', 'all']:
        # TODO: Add interpretation tips
        # TODO: Dataframe output for users who want to retain info
        # appends #
        # (1) title of method section
        result.append(f"<<______DISTRIBUTION ANALYSIS______>>")
        # (2) subtitle
        result.append(f"✎ Overview of Results*\n")

        # main operation: descriptive stats, skewness, kurtosis, normality testing
        for variable_name in numerical_variables:

            # modify data for this analysis: no NAs
            data = df[variable_name].dropna()

            # calculate descriptive stats
            mean = data.mean()
            median = data.median()
            mode = data.mode().tolist()
            var_min, var_max = data.min(), data.max()
            variance = data.var()
            std_dev = data.std()

            # calculate skewness, kurtosis and tests for normality
            skewness = skew(data)
            kurt = kurtosis(data)
            shapiro_stat, shapiro_p = shapiro(data)
            anderson_stat = anderson(data)

            # construct console output
            result.append(f"< Distribution Summary for: ['{variable_name}'] >\n")
            result.append(f"➡ Min: {var_min:.2f}\n➡ Max: {var_max:.2f}\n➡ Mean: {mean:.2f}\n➡ Median: {median:.2f}\n➡ Mode(s): {mode}")
            result.append(f"➡ Variance: {variance:.2f}\n➡ Standard Deviation: {std_dev:.2f}")
            result.append(f"➡ Skewness: {skewness:.2f}\n➡ Kurtosis: {kurt:.2f}")
            result.append(f"\n★ Shapiro-Wilk Test for Normality:\n   ➡ p-value = {shapiro_p:.4f} (Normal distribution suggested if p > 0.05)")
            result.append(f"\n★ Anderson-Darling Test for Normality:\n   ➡ statistic = {anderson_stat.statistic:.4f}\n   ➡ significance levels = {anderson_stat.significance_level}\n   ➡ critical values = {anderson_stat.critical_values}\n")

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

    elif output.lower() == 'return':
        # normal functionality of output: return
        if method.lower() == 'all':
            return combined_result

        # extended functionality of output: return (method-specific)
        if method.lower() == 'outliers_zscore':
            return outliers_z_dict, outliers_z_df

        if method.lower() == 'outliers_iqr':
            return outliers_iqr_dict, outliers_iqr_df

    else:
        raise ValueError("Invalid output method. Choose 'print' or 'return'.")


# smoke tests
pengu = pd.read_csv('./datasets/penguins.csv')
cols = [
    'culmen_length_mm', 'culmen_depth_mm',
    'flipper_length_mm', 'body_mass_g'
]
explore_num(pengu, cols, method='distribution')
# outlier_dict, outlier_df = explore_num(pengu, cols, method='outliers_zscore')
