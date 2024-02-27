import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder
)
from datasafari import explore_cat


# main function: transform_cat
def transform_cat(df: pd.DataFrame, categorical_variables: list, method: str = 'all', output: str = 'print'):
    # TODO: preprocess categorical data to make: uniform (e.g. Education column has University and university, etc.)
    # TODO: easily create ordinal categorical variables
    # TODO: easily create nominal categorical variables
    # TODO: ordinal encoding with overwrite or no overwrite
    # TODO: one hot encoding

    if method.lower() == 'encode_onehot':
        print(f"< ONE-HOT ENCODING: {categorical_variables} >\n✎ Note: Please make sure your data is cleaned.\n☻ Tip: You can use explore_cat() to check your data!\n")

        # initialize df to work with
        transformed_df = df

        # create an instance of OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse=False)

        # encode all variables at once for efficiency
        encoded_data = onehot_encoder.fit_transform(df[categorical_variables])

        # convert the encoded data to a DataFrame
        encoded_columns = pd.DataFrame(
            encoded_data,
            columns=onehot_encoder.get_feature_names_out(categorical_variables)
        )

        # reset indices of both DataFrames to ensure they align when concatenating
        transformed_df.reset_index(drop=True, inplace=True)
        encoded_columns.reset_index(drop=True, inplace=True)

        # concatenate the original DataFrame (without the categorical columns) with the encoded columns
        transformed_df = pd.concat([transformed_df.drop(columns=categorical_variables), encoded_columns], axis=1)

        # inform user
        print(f"✔ Created a transformed dataframe:\n{transformed_df.head()}\n\n✔ Created a dataframe with only the encoded columns:\n{encoded_columns.head()}\n\n☻ HOW TO: transformed_df, encoded_columns = transform_cat(yourdf, yourcolumns, method='encode_onehot'\n")

        # sanity check
        print(f"< SANITY CHECK >\n  ➡ Shape of original df: {df.shape}\n  ➡ Shape of transformed df: {transformed_df.shape}\n")

        return transformed_df, encoded_columns


# smoke tests
pengu = pd.read_csv('./datasets/penguins.csv')

transformed_df1, encoded_columns1 = transform_cat(pengu, ['sex'], method='encode_onehot')
