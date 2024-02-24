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

    if method.lower() in ['outliers_zscore', 'all']:

        # definitions #
        # (1) z-score threshold for outlier detection
        threshold_z = 3

        # temporary df used for calculation
        df_z = df
        outliers_z = {}

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

            # Drop z-score column
            # penguins_clean = df_z.drop(columns=[z_col])

        # sanity check - oid should be a dictionary of:
        # key: column name in df
        # value: list of values that have z-scores higher than 3
        # the idea is that we don't burden the df out of the for-loop
        # with new z-score columns, but extract which rows are outliers
        result.append(f"{outliers_z.items()}")

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

