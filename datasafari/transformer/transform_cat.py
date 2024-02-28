import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from scipy.cluster.hierarchy import linkage, fcluster
import Levenshtein as lev
from collections import Counter
from datasafari import explore_cat


# main function: transform_cat
def transform_cat(df: pd.DataFrame, categorical_variables: list, method: str, na_placeholder: str = 'Unknown', abbreviation_map: dict = None):
    # TODO: encode_ordinal
    # TODO: encode_frequency
    # TODO: encode_target
    # TODO: encode_binary
    # TODO: hashing
    # TODO: Add ValueError capture for method on transform_cat (if method.lower() in [all methods..]
    # TODO: Add ValueError capture for method on explore_df (if method.lower() in [all methods..]
    # TODO: Add ValueError capture for method on explore_cat (if method.lower() in [all methods..]
    # TODO: Add ValueError capture for method on explore_num (if method.lower() in [all methods..]

    if method.lower() == 'uniform_simple':
        print(f"< UNIFORM SIMPLE TRANSFORMATION* >")
        print(f" This method applies basic but effective transformations to make categorical data uniform:")
        print(f"  ✔ Lowercases all text to fix capitalization inconsistencies.")
        print(f"  ✔ Trims leading and trailing whitespaces for cleanliness.")
        print(f"  ✔ Removes special characters to standardize text.")
        print(f"  ✔ Fills missing values with a placeholder to maintain data integrity. (use na_placeholder = '...', default 'Unknown')")

        transformed_df = df.copy()
        uniform_columns = pd.DataFrame()

        for variable in categorical_variables:
            # Convert to handle missing values, lowercase, strip, and remove special characters
            transformed_column = (
                transformed_df[variable]
                .astype(str)
                .fillna(na_placeholder)  # Customize this placeholder as needed
                .str.lower()
                .str.strip()
                .str.replace('[^a-zA-Z0-9\s]', '', regex=True)
            )
            transformed_df[variable] = transformed_column
            uniform_columns = pd.concat([uniform_columns, transformed_column], axis=1)

            print(f"\n['{variable}'] Category Transformation\n")
            print(f"Categories BEFORE transformation ({len(df[variable].unique())}): {df[variable].unique()}\n")
            print(f"Categories AFTER transformation ({len(transformed_df[variable].unique())}): {transformed_df[variable].unique()}\n")

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the uniform columns:\n{uniform_columns.head()}\n")
        print("☻ HOW TO: To catch the df's use - `transformed_df, uniform_columns = transform_cat(your_df, your_columns, method='uniform_simple')`.\n")
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original df: {df.shape}")
        print(f"  ➡ Shape of transformed df: {transformed_df.shape}\n")
        print("* For more advanced and nuanced data cleaning, consider using `uniform_smart`.")

        return transformed_df, uniform_columns

    if method.lower() == 'uniform_smart':
        print(f"< UNIFORM SMART TRANSFORMATION* >")
        print(f" This method leverages advanced data cleaning techniques for categorical variables, enhancing uniformity across your dataset:")
        print(f"  ✔ Utilizes the `uniform_simple` method for initial preprocessing steps.")
        print(f"  ✔ Employs Levenshtein distance to evaluate textual similarity among categories.")
        print(f"  ✔ Applies hierarchical clustering to group similar categories together.")
        print(f"  ✔ Selects the most representative category within each cluster to ensure data consistency.")
        print(f"  ✔ Fills missing values with a placeholder to maintain data integrity. (default 'Unknown', customize with na_placeholder = '...')\n")

        transformed_df = df.copy()
        uniform_columns = pd.DataFrame()

        for variable in categorical_variables:
            # preprocess categories using uniform_simple method
            categories = (
                transformed_df[variable]
                .astype(str)
                .fillna(na_placeholder)  # Customize this placeholder as needed
                .str.lower()
                .str.strip()
                .str.replace('[^a-zA-Z0-9\s]', '', regex=True)
            )
            unique_categories = categories.unique()

            # calculate the similarity matrix using Levenshtein distance
            dist_matrix = np.array([[lev.distance(w1, w2) for w1 in unique_categories] for w2 in unique_categories])
            max_dist = np.max(dist_matrix)
            sim_matrix = 1 - dist_matrix / max_dist if max_dist != 0 else np.zeros_like(dist_matrix)

            # perform hierarchical clustering on the similarity matrix
            Z = linkage(sim_matrix, method='complete')
            cluster_labels = fcluster(Z, t=0.5, criterion='distance')

            # assign each category to a cluster representative
            representatives = {}
            for cluster in np.unique(cluster_labels):
                index = np.where(cluster_labels == cluster)[0]
                cluster_categories = unique_categories[index]
                # select the most frequent category within each cluster as the representative
                representative = max(cluster_categories, key=lambda x: (categories == x).sum())
                representatives[cluster] = representative

            # map each category in the original dataframe to its cluster representative
            category_to_representative = {cat: representatives[cluster_labels[i]] for i, cat in enumerate(unique_categories)}
            transformed_df[variable] = categories.map(category_to_representative)

            # add transformed column to uniform_columns DataFrame
            uniform_columns = pd.concat([uniform_columns, transformed_df[[variable]]], axis=1)

            print(f"\n['{variable}'] Category Transformation\n")
            print(f"Categories BEFORE transformation ({len(df[variable].unique())}): {df[variable].unique()}\n")
            print(f"Categories AFTER transformation ({len(transformed_df[variable].unique())}): {transformed_df[variable].unique()}\n")

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the uniform columns:\n{uniform_columns.head()}\n")
        print("☻ HOW TO: To catch the df's use - `transformed_df, uniform_columns = transform_cat(your_df, your_columns, method='uniform_smart')`.\n")
        print("< SANITY CHECK >")
        print(f"  ➡ Shape of original df: {df.shape}")
        print(f"  ➡ Shape of transformed df: {transformed_df.shape}\n")
        print("* Consider `uniform_simple` for basic data cleaning needs or when processing large datasets where computational efficiency is a concern.")

        return transformed_df, uniform_columns

    if method.lower() == 'encode_onehot':
        print(f"< ONE-HOT ENCODING: {categorical_variables} >\n✎ Note: Please make sure your data is cleaned.\n☻ Tip: You can use explore_cat() to check your data!\n")

        # initialize df to work with
        transformed_df = df.copy()

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



# create simple test dataset for smoke tests: nonuniform_df #
nonuniform_data = {
    'Category': [
        'Student', 'student', 'STUDENT', 'St!UdE$nT',  # Variations of "student"
        'high school', 'High School', 'high   school', 'highschool', 'hgh schl',  # Variations of "high school"
        'university', 'University', 'UNIVERSITY', 'universty',  # Variations of "university"
        'college', 'College', 'COLLEGE', 'collg'  # Variations of "college"
    ]
}
nonuniform_data['Value'] = np.random.randint(1, 100, size=len(nonuniform_data['Category']))
nonuniform_df = pd.DataFrame(nonuniform_data)


# 'uniform_...' tests #

# uniform_simple
simple_transformed_df, simple_transformed_cols = transform_cat(nu_df, ['Category'], method='uniform_simple')

# uniform_smart
smart_transformed_df, smart_transformed_cols = transform_cat(nonuniform_df, ['Category'], method='uniform_smart')

# uniform_mapping
abbreviation_map = {
    'Category': {
        'high   school': 'high school',
        'hgh schl': 'high school'
    }
}
final_transformed_df, final_transformed_cols = transform_cat(smart_transformed_df, ['Category'], method='uniform_mapping', abbreviation_map=abbreviation_map)


# 'encode_...' tests #

# encode_onehot
onehot_encoded_df, onehot_encoded_cols = transform_cat(final_transformed_df, ['Category'], method='encode_onehot')
