import pandas as pd


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
    if method.lower() in ['outliers_iqr', 'all']:

        # appends #
        # (1) title of method section
        result.append(f"<<______OUTLIERS - IQR METHOD______>>\n")
        # (2) suitability tip
        result.append(f"Tip: The IQR method is robust against extreme values, ideal for identifying outliers in skewed distributions by focusing on the data's middle 50%.")

    if method.lower() in ['outliers_zscore', 'all']:

        # definitions #
        # (1) z-score threshold for outlier detection
        threshold_z = 3

        # appends #
        # (1) title of method section
        result.append(f"<<______OUTLIERS - Z-SCORE METHOD______>>\n")
        # (2) suitability tip
        result.append(f"Tip: The Z-Score method excels at identifying outliers in data with a distribution close to normal, highlighting values far from the mean.\n")
        # (2) outlier df info
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

            # classify outlier values
            outlier_z_values = (
                df[df[z_col].abs() > threshold_z][variable_name].tolist()
            )

            # save results to outliers_z_dict
            outliers_z_dict[variable_name] = outlier_z_values

            # save resulting rows to outliers_z_df
            outlier_rows = df[df[variable_name].isin(outliers_z_dict[variable_name])]
            outliers_z_df = pd.concat([outliers_z_df, outlier_rows], ignore_index=False)

            # clean the original df
            df.drop(z_col, axis=1, inplace=True)

            # informative stats about outliers #
            outlier_count = len(outlier_z_values)
            # conditional output string format and stats calculations
            title = f"< Results for ['{variable_name}'] >\n"
            result.append(title)
            if outlier_count == 0:
                stats = f"➡ Number of outliers: {outlier_count}\n➡ Min: -\n➡ Max: -\n➡ Mean: -"
                row_indices = f"➡ Location of outliers in your df (indices): -\n"
                result.append(stats)
                result.append(row_indices)
            else:
                outlier_min = min(outlier_z_values)
                outlier_max = max(outlier_z_values)
                outlier_mean = sum(outlier_z_values) / len(outlier_z_values)
                stats = f"➡ Number of outliers: {outlier_count}\n➡ Min: {outlier_min}\n➡ Max: {outlier_max}\n➡ Mean: {outlier_mean}"
                row_indices = f"➡ Location of outliers in your df (indices):\n{outlier_rows.index.tolist()}\n"
                result.append(stats)
                result.append(row_indices)

        # appends (continued) #
        # (6-9) method='outlier_z' info
        result.append(f"✎ * NOTE: If method='outliers_zscore', aside from the overview above, the function RETURNS:")
        result.append(f"■ 1 - Dictionary: key=variable name, value=list of outlier values for that row")
        result.append(f"■ 2 - Dataframe: Rows from the original df that were classified as outliers. (preserved index)")
        result.append(f"☻ HOW TO: dict, df = explore_num(yourdf, yourlist, method='outliers_zscore')")

    # Combine all results
    combined_result = "\n".join(result)

    if output.lower() == 'print':
        print(combined_result)

        if method.lower() == 'outliers_zscore':
            return outliers_z_dict, outliers_z_df

    elif output.lower() == 'return':

        if method.lower() == 'outliers_zscore':
            return outliers_z_dict, outliers_z_df

    else:
        raise ValueError("Invalid output method. Choose 'print' or 'return'.")


# smoke tests
pengu = pd.read_csv('./datasets/penguins.csv')
cols = [
    'culmen_length_mm', 'culmen_depth_mm',
    'flipper_length_mm', 'body_mass_g'
]
explore_num(pengu, cols)
outlier_dict, outlier_df = explore_num(pengu, cols, method='outlier_z')
