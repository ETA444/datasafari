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

        # create an instance of OneHotEncoder
        onehot_encoder = OneHotEncoder()

        # encode the selected variables
        for variable in categorical_variables:
            print(f"< ONE-HOT ENCODING: ['{variable}'] >\n✎ Note: Please make sure your data is cleaned.\n☻ Tip: You can use explore_cat() to check your data!\n")
            transformed_df = df

            # fit and transform the data
            encoded_values = onehot_encoder.fit_transform(transformed_df[[variable]])

            # convert the sparse matrix to a DataFrame and append to ci
            encoded_columns = pd.DataFrame(
                encoded_values.toarray(),
                columns=onehot_encoder.get_feature_names_out([variable])
            )

            # Reset indices of both DataFrames
            transformed_df.reset_index(drop=True, inplace=True)
            encoded_columns.reset_index(drop=True, inplace=True)

            # append the new columns to df
            transformed_df = pd.concat([transformed_df, encoded_columns], axis=1)

            # drop the original column
            transformed_df.drop(columns=[variable], inplace=True)

            # inform user
            print(f"✔ Created a transformed dataframe:\n{transformed_df.head()}\n\n✔ Created a dataframe with only the encoded columns:\n{encoded_columns.head()}\n\n☻ HOW TO: transformed_df, encoded_columns = transform_cat(yourdf, yourcolumns, method='encode_onehot'\n")

            # sanity check
            print(f"< SANITY CHECK >\n  ➡ Shape of original df: {df.shape}\n  ➡ Shape of transformed df: {transformed_df.shape}\n")

            # return transformed_df to user
            return transformed_df, encoded_columns


# smoke tests
pengu = pd.read_csv('./datasets/penguins.csv')

transformed_df1, encoded_columns1 = transform_cat(pengu, ['sex'], method='encode_onehot')
