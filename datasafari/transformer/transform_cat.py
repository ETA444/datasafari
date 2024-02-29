import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)
from scipy.cluster.hierarchy import linkage, fcluster
import Levenshtein as lev
from category_encoders import BinaryEncoder


# main function: transform_cat
def transform_cat(df: pd.DataFrame, categorical_variables: list, method: str, na_placeholder: str = 'Unknown', abbreviation_map: dict = None, ordinal_map: dict = None, target_variable: str = None):

    if method.lower() == 'uniform_simple':
        print(f"< UNIFORM SIMPLE TRANSFORMATION* >")
        print(f" This method applies basic but effective transformations to make categorical data uniform:")
        print(f"  ✔ Lowercases all text to fix capitalization inconsistencies.")
        print(f"  ✔ Trims leading and trailing whitespaces for cleanliness.")
        print(f"  ✔ Removes special characters to standardize text.")
        print(f"  ✔ Fills missing values with a placeholder to maintain data integrity. (use na_placeholder = '...', default 'Unknown')")

        # initialize dataframe to work with
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

        # initialize dataframe to work with
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

    if method.lower() == 'uniform_mapping' and abbreviation_map:
        print(f"< MANUAL CATEGORY MAPPING >")
        print(" This method allows for manual mapping of categories to address specific cases:")
        print("  ✔ Maps categories based on user-defined rules.")
        print("  ✔ Useful for stubborn categories that automated methods can't uniformly transform.")
        print("✎ Note: Ensure your mapping dictionary is comprehensive for the best results.\n")

        # initialize dataframe to work with
        transformed_df = df.copy()
        uniform_columns = pd.DataFrame()

        for variable in categorical_variables:
            if variable in abbreviation_map:
                # apply mapping
                transformed_df[variable] = transformed_df[variable].map(lambda x: abbreviation_map[variable].get(x, x))
                uniform_columns = pd.concat([uniform_columns, transformed_df[[variable]]], axis=1)

                print(f"\n['{variable}'] Category Mapping\n")
                print(f"Categories BEFORE mapping ({len(df[variable].unique())}): {df[variable].unique()}\n")
                print(f"Categories AFTER mapping ({len(transformed_df[variable].unique())}): {transformed_df[variable].unique()}\n")

        print("< SANITY CHECK >")
        print(f"  ➡ Original dataframe shape: {df.shape}")
        print(f"  ➡ Transformed dataframe shape: {transformed_df.shape}\n")

        return transformed_df, uniform_columns

    if method.lower() == 'encode_onehot':
        print(f"< ONE-HOT ENCODING TRANSFORMATION >")
        print(f" This method converts categorical variables into a form that could be provided to ML algorithms to do a better job in prediction:")
        print(f"  ✔ Converts each category value into a new column and assigns a 1 or 0 (notation for true/false).")
        print(f"  ✔ Ensures the data is ready for machine learning models without assuming any ordinal relationship.")
        print(f"  ✔ Helps to tackle the issue of 'curse of dimensionality' in a controlled manner.")
        print(f"✎ Note: Before encoding, ensure data uniformity and cleanliness.\n☻ Tip: Use `uniform_simple` or `uniform_smart` from `transform_cat()` for advanced categorical data cleaning.\n")

        # initialize dataframe to work with
        transformed_df = df.copy()

        # create an instance of one hot encoder
        onehot_encoder = OneHotEncoder(sparse=False)

        # encode all variables at once for efficiency
        encoded_data = onehot_encoder.fit_transform(df[categorical_variables])

        # convert the encoded data to a dataframe
        encoded_columns = pd.DataFrame(
            encoded_data,
            columns=onehot_encoder.get_feature_names_out(categorical_variables)
        )

        # reset indices of both dataframes to ensure they align when concatenating
        transformed_df.reset_index(drop=True, inplace=True)
        encoded_columns.reset_index(drop=True, inplace=True)

        # concatenate the original dataframe (minus the categorical columns) with the encoded columns
        transformed_df = pd.concat([transformed_df.drop(columns=categorical_variables), encoded_columns], axis=1)

        print(f"✔ New transformed dataframe with one-hot encoded variables:\n{transformed_df.head()}\n")
        print(f"✔ Separate dataframe with only the one-hot encoded columns:\n{encoded_columns.head()}\n")
        print("☻ HOW TO: To catch the df's use - `transformed_df, encoded_columns = transform_cat(your_df, your_columns, method='encode_onehot')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Original dataframe shape: {df.shape}")
        print(f"  ➡ Transformed dataframe shape: {transformed_df.shape}\n")

        return transformed_df, encoded_columns

    if method.lower() == 'encode_ordinal' and ordinal_map:
        print(f"< ORDINAL ENCODING TRANSFORMATION >")
        print(f" This method assigns an integer to each category value based on the provided ordinal order.")
        print(f"✎ Note: Ensure the provided ordinal map correctly reflects the desired order of categories for each variable.")
        print("☻ Tip: An ordinal map dictionary looks like this: {'your_variable': ['level1', 'level2', 'level3'], ...}\n")

        # initialize dataframe to work with
        transformed_df = df.copy()
        encoded_columns = pd.DataFrame()

        # encode each variable according to the provided ordinal map
        for variable, order in ordinal_map.items():
            if variable in categorical_variables:
                # Prepare data for OrdinalEncoder
                data = transformed_df[[variable]].apply(lambda x: pd.Categorical(x, categories=order, ordered=True))
                transformed_df[variable] = data.apply(lambda x: x.cat.codes)

                # Keep track of the newly encoded columns
                encoded_columns = pd.concat([encoded_columns, transformed_df[[variable]]], axis=1)

                print(f"✔ '{variable}' encoded based on the specified order: {order}\n")
            else:
                print(f"⚠️ '{variable}' specified in `ordinal_map` was not found in `categorical_variables` and has been skipped.\n")

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only the ordinal encoded columns:\n{encoded_columns.head()}\n")
        print("☻ HOW TO - To catch the df's use: `transformed_df, encoded_columns = transform_cat(your_df, your_columns, method='encode_ordinal', ordinal_map=your_ordinal_map)`.\n")
        print("< SANITY CHECK >")
        print(f"  ➡ Original dataframe shape: {df.shape}")
        print(f"  ➡ Transformed dataframe shape: {transformed_df.shape}\n")

        return transformed_df, encoded_columns

    if method.lower() == 'encode_freq':
        print(f"< FREQUENCY ENCODING TRANSFORMATION >")
        print(f" This method transforms categorical variables based on the frequency of each category.")
        print(f"✎ Note: Frequency encoding helps to retain the information about the category's prevalence.")
        print(f"☻ Tip: Useful for models where the frequency significance of categories impacts the prediction.\n")

        # initialize dataframe to work with
        transformed_df = df.copy()
        encoded_columns = pd.DataFrame()

        for variable in categorical_variables:
            # calculate the frequency of each category
            frequency_map = transformed_df[variable].value_counts().to_dict()

            # map the frequencies to the original dataframe
            transformed_df[variable] = transformed_df[variable].map(frequency_map)
            encoded_columns = pd.concat([encoded_columns, transformed_df[[variable]]], axis=1)
            print(f"✔ '{variable}' has been frequency encoded.\n")

        print(f"✔ New transformed dataframe:\n{transformed_df.head()}\n")
        print(f"✔ Dataframe with only frequency encoded columns:\n{encoded_columns.head()}\n")
        print("☻ HOW TO - to catch the df's: `transformed_df, encoded_columns = transform_cat(your_df, your_columns, method='encode_freq')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Original dataframe shape: {df.shape}")
        print(f"  ➡ Transformed dataframe shape: {transformed_df.shape}\n")

        return transformed_df, encoded_columns

    if method.lower() == 'encode_target' and target_variable:
        print(f"< TARGET ENCODING TRANSFORMATION* >")
        print(f" This method encodes categorical variables based on the mean of the target variable for each category.")
        print(f"✎ Note: Target encoding captures the 'effect' of each category on the target variable.")
        print(f"☻ Tip: To prevent data leakage and overfitting, apply target encoding within a cross-validation loop, ensuring it's computed separately for each fold.\n")

        # initialize dataframe to work with
        transformed_df = df.copy()
        encoded_columns = pd.DataFrame()

        for variable in categorical_variables:
            # calculate the mean of the target variable for each category
            target_means = transformed_df.groupby(variable)[target_variable].mean()

            # map the computed means to the original categories
            transformed_df[variable] = transformed_df[variable].map(target_means)
            encoded_columns = pd.concat([encoded_columns, transformed_df[[variable]]], axis=1)

            print(f"✔ '{variable}' has been target encoded with respect to '{target_variable}'.\n")

        print(f"✔ New transformed dataframe with target encoded variables:\n{transformed_df.head()}\n")
        print(f"✔ Separate dataframe with only the target encoded columns:\n{encoded_columns.head()}\n")
        print("☻ HOW TO - to catch the df's: `transformed_df, encoded_columns = transform_cat(your_df, your_columns, method='encode_target', target_variable='your_target_variable')`.\n")

        # Sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Original dataframe shape: {df.shape}")
        print(f"  ➡ Transformed dataframe shape: {transformed_df.shape}\n")

        return transformed_df, encoded_columns

    if method.lower() == 'encode_binary':
        print(f"< BINARY ENCODING TRANSFORMATION >")
        print(f" This method transforms categorical variables into binary columns, significantly reducing dimensionality for high cardinality features.")
        print(f"  ✔ Efficiently handles categories by representing them with binary codes.")
        print(f"  ✔ Reduces dataset size and model complexity compared to one-hot encoding.")
        print(f"✎ Note: Ideal for categorical variables with many unique categories.\n☻ Tip: For categories with limited unique values, consider if binary encoding aligns with your data strategy.\n")

        # initialize dataframe to work with
        transformed_df = df.copy()

        # create an instance of BinaryEncoder
        binary_encoder = BinaryEncoder(cols=categorical_variables)

        # keep a copy of the original columns, including the categorical ones that will be encoded
        original_columns = transformed_df.columns.tolist()

        # fit and transform the data
        transformed_df = binary_encoder.fit_transform(transformed_df)

        # determine the new columns by excluding the original columns
        new_columns = [col for col in transformed_df.columns if col not in original_columns]
        # note: this assumes the BinaryEncoder removes the original categorical columns

        # extract the newly created binary encoded columns for user reference
        encoded_columns = transformed_df[new_columns]

        print(f"✔ New transformed dataframe with binary encoded variables:\n{transformed_df.head()}\n")
        print(f"✔ Separate dataframe with only the binary encoded columns:\n{encoded_columns.head()}\n")
        print("☻ HOW TO - to catch the new dfs: `transformed_df, encoded_columns = transform_cat(your_df, your_columns, method='encode_binary')`.\n")

        # sanity check
        print("< SANITY CHECK >")
        print(f"  ➡ Original dataframe shape: {df.shape}")
        print(f"  ➡ Transformed dataframe shape: {transformed_df.shape}\n")

        return transformed_df, encoded_columns

    # Error Handling #
    # ValueError if method not in valid_methods
    valid_methods = [
        'uniform_simple', 'uniform_smart', 'uniform_mapping',  # uniform family
        'encode_onehot', 'encode_ordinal', 'encode_freq', 'encode_target', 'encode_binary'  # encode family
    ]
    if method.lower() not in valid_methods:
        raise ValueError(f"Invalid method. You can choose from: {valid_methods}")

    # Methods for transform_cat
    # DONE! TODO: encode_ordinal
    # DONE! TODO: encode_frequency
    # DONE! TODO: encode_target
    # DONE! TODO: encode_binary

    # TODO: Docstring for transform_cat

    # Error Handling
    # DONE! TODO: Add ValueError capture for method on transform_cat (if method.lower() in [all methods..]
    # TODO: Add ValueError capture for method on explore_df (if method.lower() in [all methods..]
    # TODO: Add ValueError capture for method on explore_cat (if method.lower() in [all methods..]
    # TODO: Add ValueError capture for method on explore_num (if method.lower() in [all methods..]
    # TODO: Add ValueError capture for other components <- for each module

    # Testing
    # TODO: Create tests/  for all

    # Documentation
    # TODO: Create official documentation


# smoke tests #


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
simple_transformed_df, simple_transformed_cols = transform_cat(nonuniform_df, ['Category'], method='uniform_simple')

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

# encode_ordinal
ordinal_map = {
    'Category': ['student', 'high school', 'college', 'university']
}

ordinal_encoded_df, ordinal_encoded_cols = transform_cat(final_transformed_df, ['Category'], method='encode_ordinal', ordinal_map=ordinal_map)

# encode_freq
freq_encoded_df, freq_encoded_cols = transform_cat(final_transformed_df, ['Category'], method='encode_freq')

# encode_target
target_encoded_df, target_encoded_cols = transform_cat(final_transformed_df, ['Category'], method='encode_target', target_variable='Value')

# encode_binary
binary_encoded_df, binary_encoded_cols = transform_cat(final_transformed_df, ['Category'], method='encode_binary')