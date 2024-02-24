# main function: explore_num
def explore_num(df: pd.DataFrame, numerical_variables: list, method: str = 'all', output: str = 'print'):
    """..."""
    result = []

    if method.lower() in ['outlier_z', 'all']:
        # initial append for title of method section
        result.append(f"<<______OUTLIERS - Z-SCORE METHOD______>>\n")

        # temporary df used for calculation
        df_z = df
        outliers_z = {}

        # get the unique values per variable in categorical_variables list
        for variable_name in numerical_variables:

            # zscore column name
            z_col = variable_name+'_zscore'

            # calculate z-score for col
            df_z[z_col] = (
                    (df_z[variable_name] - df_z[variable_name].mean())
                    / df_z[variable_name].std()
            )

            # threshold for outlier (z-score > 3 or < -3)
            threshold = 3

            # identify outliers
            oid_values = (
                df_z[
                    (df_z[z_col].abs() > threshold)
                ][variable_name].tolist()
            )

            # save results to oid dictionary
            outliers_z[variable_name] = oid_values

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
    elif output.lower() == 'return':
        return combined_result
    else:
        raise ValueError("Invalid output method. Choose 'print' or 'return'.")

