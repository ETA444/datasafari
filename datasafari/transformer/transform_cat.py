from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder
)
from scipy.cluster.hierarchy import linkage, fcluster
import Levenshtein as lev
from category_encoders import BinaryEncoder
from datasafari.evaluator.evaluate_dtype import evaluate_dtype


def transform_cat(
        df: pd.DataFrame,
        categorical_variables: List[str],
        method: str,
        na_placeholder: str = 'Unknown',
        abbreviation_map: Optional[Dict[str, Dict[str, str]]] = None,
        ordinal_map: Optional[Dict[str, List[str]]] = None,
        target_variable: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    **Transform categorical variables in a DataFrame through a range of encoding options and basic to advanced machine learning-based methods for uniform data cleaning.**

    This is a versatile tool designed for comprehensive transformation and encoding of categorical data in a DataFrame. It accommodates everything from simple text standardization to sophisticated category consolidation using ML and various encoding schemes, catering to both nominal and ordinal data.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the categorical data to transform.

    categorical_variables : list
        A list of strings representing the names of the categorical columns to be transformed.

    method : str
        The method to use for transforming the categorical variables.
            - ``'uniform_simple'`` Basic cleaning transformations to standardize text, such as lowercase conversion, whitespace trimming, and special character removal. Also fills missing values with a specified placeholder.
            - ``'uniform_smart'`` Advanced cleaning that leverages Levenshtein distance for textual similarity and hierarchical clustering to group and normalize similar categories. Builds on 'uniform_simple' preprocessing steps.
            - ``'uniform_mapping'`` Allows for manual mapping of categories based on user-defined rules to handle specific cases that automated methods might not cover.
            - ``'encode_onehot'`` Converts categories into binary columns for each category. Suitable for nominal data where no ordinal relationship exists.
            - ``'encode_ordinal'`` Maps categories to an integer array based on the order defined in `ordinal_map`. Suitable for ordinal data where the order of categories is important.
            - ``'encode_freq'`` Transforms categories based on the frequency of each category, replacing the category name with its frequency count.
            - ``'encode_target'`` Encodes categories based on the mean of the target variable for each category. This method should be used cautiously to avoid data leakage and is recommended to be applied within a cross-validation loop.
            - ``'encode_binary'`` Utilizes binary encoding to transform categories into binary columns, reducing dimensionality and dataset size compared to one-hot encoding. Ideal for high cardinality features.

    na_placeholder : str, optional, default: 'Unknown'
        The placeholder value to use for missing values during transformations.

    abbreviation_map : dict, optional, default: None
        A dictionary specifying manual mappings for categories, used with the 'uniform_mapping' method.
            - Each key should be the name of a categorical variable, and its value should be another dictionary mapping original category values to their new values.

    ordinal_map : dict, optional, default: None
        A dictionary specifying the order of categories for ordinal encoding, used with the 'encode_ordinal' method.
            - Each key should be the name of a categorical variable, and its value should be a list of categories in the desired order. This method treats the order of categories as meaningful and encodes them as integers based on the provided order.

    target_variable : str, optional, default: None
        The name of the target variable for target encoding. Used with the 'encode_target' method.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Original DataFrame with transformed categorical variables.
        - A DataFrame containing only the transformed columns.

    Raises:
    -------
    TypeErrors:
        - If `df` is not a pandas DataFrame.
        - If `categorical_variables` is not a list or contains elements that are not strings.
        - If `method`, `na_placeholder`, or `target_variable` is not a strings.
        - If `abbreviation_map` or `ordinal_map`  is not a dictionary.

    ValueErrors:
        - If the input DataFrame is empty.
        - If 'categorical_variables' list is empty.
        - If variables provided through 'categorical_variables' are not categorical variables.
        - If any variable specified in `categorical_variables` is not found in the DataFrame's columns.
        - If `method` is not one of the valid options.
        - If `method` is 'encode_ordinal' and `ordinal_map` is not provided.
        - If `method` is 'encode_target' and `target_variable` is not provided.
        - If `method` is 'uniform_mapping' and `abbreviation_map` is not provided.
        - If `target_variable` is specified but not found in the DataFrame's columns.
        - If keys specified in `abbreviation_map` or `ordinal_map` are not found in the DataFrame's columns.

    Examples:
    ---------
    Import necessary libraries and generate a DataFrame for examples:

    >>> import datasafari
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
        ...     'Category': ['Student', 'student', 'STUDENT', 'StUdEnT', 'high school', 'High School', 'high   school', 'hgh schl'],
        ...     'Target': np.random.randint(0, 2, 8)
        ... })

    Apply ``'uniform_simple'`` method to clean up ```Category`` column:

    >>> transformed_df, uniform_simple_cols = transform_cat(df, ['Category'], method='uniform_simple')

    Apply ``'uniform_smart'`` method to clean up more complex issues in the column using ML techniques:

    >>> transformed_df, uniform_smart_cols = transform_cat(transformed_df, ['Category'], method='uniform_smart')
    >>> # Note: 'uniform_smart' already has 'uniform_simple' built-in, however to save resources you can run smart only if needed.

    Apply ``'uniform_mapping'`` method with a custom abbreviation map to clean up the more stubborn issues that ``uniform_smart`` did not catch:

    >>> abbreviation_map = {'Category': {'hgh schl': 'high school'}}
    >>> transformed_df, uniform_mapped_cols = transform_cat(transformed_df, ['Category'], method='uniform_mapping', abbreviation_map=abbreviation_map)

    Using the various encoding methods now that the data is clean:

    >>> # Apply 'encode_onehot' method:
    >>> transformed_onehot_df, onehot_cols = transform_cat(transformed_df, ['Category'], method='encode_onehot')
    ...
    >>> # Apply 'encode_ordinal' method with a custom ordinal map:
    >>> ordinal_map = {'Category': ['student', 'high school']}
    >>> transformed_ordinal_df, ordinal_cols = transform_cat(transformed_df, ['Category'], method='encode_ordinal', ordinal_map=ordinal_map)
    ...
    >>> # Apply 'encode_freq' method:
    >>> transformed_freq_df, freq_cols = transform_cat(transformed_df, ['Category'], method='encode_freq')
    ...
    >>> # Apply 'encode_target' method with a specified target variable:
    >>> transformed_target_df, target_cols = transform_cat(transformed_df, ['Category'], method='encode_target', target_variable='Target')
    ...
    >>> # Apply 'encode_binary' method:
    >>> transformed_binary_df, binary_cols = transform_cat(transformed_df, ['Category'], method='encode_binary')

    Notes:
    ------
    **`uniform_simple` Method:**
    This method provides a foundational approach to cleaning categorical data by implementing several straightforward transformations to standardize and simplify the text data. Here’s a breakdown of the steps involved:

        1. **Lowercase Conversion**: All characters in the text are converted to lowercase to eliminate inconsistencies caused by varied capitalizations.

        2. **Whitespace Trimming**: Leading and trailing spaces are removed from each string to ensure cleanliness and uniformity in the text data.

        3. **Special Characters Removal**: Non-alphanumeric characters are removed to standardize the text and reduce noise in the data. This step is crucial for preparing data for machine learning models, which may be sensitive to such variations.

        4. **Missing Values Handling**: Missing values are replaced with a specified placeholder, defaulting to 'Unknown'. This ensures that no data point is lost due to absence of information, and helps maintain the integrity of the dataset during further transformations.

    **`uniform_smart` Method:**
    Building on the principles of `uniform_simple`, the `uniform_smart` method incorporates advanced techniques to address more complex variations in text data that simple transformations might miss:

        1. **Initial Preprocessing**: Executes all steps of `uniform_simple` to prepare the data, setting a standardized baseline for further processing.

        2. **Textual Similarity Evaluation**: Utilizes the Levenshtein distance, a measure of the difference between two strings, to quantify the similarity between categories. This metric helps identify and group textually similar categories, even if they are not exactly the same.

        3. **Hierarchical Clustering**: Applies hierarchical clustering to the similarity matrix generated from the Levenshtein distances. This statistical method groups categories based on their textual closeness, which allows for the aggregation of variations of the same category.

        4. **Cluster Representative Selection**: Within each identified cluster, a representative category is chosen to stand for all the categories within that cluster. Typically, the most frequent category within the cluster is selected as the representative, ensuring that the most common terminology is used in the dataset.

        5. **Category Normalization**: Each original category is mapped to its cluster representative, normalizing the dataset by reducing the variation due to synonyms, misspellings, and other irregularities.
    """

    # Error-Handling #

    # TypeErrors
    # Check if 'df' is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("transform_cat(): The 'df' parameter must be a pandas DataFrame.")

    # Check if 'categorical_variables' is a list
    if not isinstance(categorical_variables, list):
        raise TypeError("transform_cat(): The 'categorical_variables' parameter must be a list of column names.")
    else:
        if not all(isinstance(var, str) for var in categorical_variables):
            raise TypeError("transform_cat(): All elements in the 'categorical_variables' list must be strings representing column names.")

    # Check if 'method' is a string
    if not isinstance(method, str):
        raise TypeError("transform_cat(): The 'method' parameter must be a string.")

    # Check if 'na_placeholder' is a string
    if not isinstance(na_placeholder, str):
        raise TypeError("transform_cat(): The 'na_placeholder' parameter must be a string.")

    # Check if 'abbreviation_map', if provided, is a dictionary
    if abbreviation_map is not None and not isinstance(abbreviation_map, dict):
        raise TypeError("transform_cat(): The 'abbreviation_map' parameter must be a dictionary if provided.")

    # Check if 'ordinal_map', if provided, is a dictionary
    if ordinal_map is not None and not isinstance(ordinal_map, dict):
        raise TypeError("transform_cat(): The 'ordinal_map' parameter must be a dictionary if provided.")

    # Check if 'target_variable', if provided, is a string
    if target_variable is not None and not isinstance(target_variable, str):
        raise TypeError("transform_cat(): The 'target_variable' parameter must be a string if provided.")

    # ValueErrors
    # Check if df is empty
    if df.empty:
        raise ValueError("explore_num(): The input DataFrame is empty.")

    # Check if list has any members
    if len(categorical_variables) == 0:
        raise ValueError("transform_cat(): The 'categorical_variables' list must contain at least one column name.")

    # Check if specified variables exist in the DataFrame
    missing_vars = [var for var in categorical_variables if var not in df.columns]
    if missing_vars:
        raise ValueError(f"transform_cat(): The following variables were not found in the DataFrame: {', '.join(missing_vars)}")

    # Check if variables are categorical
    categorical_types = evaluate_dtype(df, categorical_variables, output='list_c')
    if not all(categorical_types):
        raise ValueError("transform_cat(): The 'categorical_variables' list must contain only names of categorical variables.")

    # Check if method is valid
    valid_methods = ['uniform_simple', 'uniform_smart', 'uniform_mapping', 'encode_onehot', 'encode_ordinal', 'encode_freq', 'encode_target', 'encode_binary']
    if method.lower() not in valid_methods:
        raise ValueError(f"transform_cat(): Invalid method '{method}'. Valid options are: {', '.join(valid_methods)}")

    # Additional checks for method-specific parameters
    if method.lower() == 'encode_ordinal' and not ordinal_map:
        raise ValueError("transform_cat(): The 'ordinal_map' parameter must be provided when using the 'encode_ordinal' method.")

    if method.lower() == 'encode_target' and not target_variable:
        raise ValueError("transform_cat(): The 'target_variable' parameter must be provided when using the 'encode_target' method.")

    if method.lower() == 'uniform_mapping' and not abbreviation_map:
        raise ValueError("transform_cat(): The 'abbreviation_map' parameter must be provided when using the 'uniform_mapping' method.")

    # Check if target_variable exists in the DataFrame for methods that require it
    if target_variable and target_variable not in df.columns:
        raise ValueError(f"transform_cat(): The target variable '{target_variable}' was not found in the DataFrame.")

    # Check if all keys in abbreviation_map are in the DataFrame columns
    if abbreviation_map and not all(key in df.columns for key in abbreviation_map.keys()):
        invalid_keys = [key for key in abbreviation_map.keys() if key not in df.columns]
        raise ValueError(f"transform_cat(): The following keys in 'abbreviation_map' were not found in the DataFrame columns: {', '.join(invalid_keys)}")

    # Check if all keys in ordinal_map are in the DataFrame columns
    if ordinal_map and not all(key in df.columns for key in ordinal_map.keys()):
        invalid_keys = [key for key in ordinal_map.keys() if key not in df.columns]
        raise ValueError(f"transform_cat(): The following keys in 'ordinal_map' were not found in the DataFrame columns: {', '.join(invalid_keys)}")

    # Main Function #
    if method.lower() == 'uniform_simple':
        print("< UNIFORM SIMPLE TRANSFORMATION* >")
        print(" This method applies basic but effective transformations to make categorical data uniform:")
        print("  ✔ Lowercases all text to fix capitalization inconsistencies.")
        print("  ✔ Trims leading and trailing whitespaces for cleanliness.")
        print("  ✔ Removes special characters to standardize text.")
        print("  ✔ Fills missing values with a placeholder to maintain data integrity. (use na_placeholder = '...', default 'Unknown')")

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
                .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
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
        print("< UNIFORM SMART TRANSFORMATION* >")
        print(" This method leverages advanced data cleaning techniques for categorical variables, enhancing uniformity across your dataset:")
        print("  ✔ Utilizes the `uniform_simple` method for initial preprocessing steps.")
        print("  ✔ Employs Levenshtein distance to evaluate textual similarity among categories.")
        print("  ✔ Applies hierarchical clustering to group similar categories together.")
        print("  ✔ Selects the most representative category within each cluster to ensure data consistency.")
        print("  ✔ Fills missing values with a placeholder to maintain data integrity. (default 'Unknown', customize with na_placeholder = '...')\n")

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
        print("< MANUAL CATEGORY MAPPING >")
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
        print("< ONE-HOT ENCODING TRANSFORMATION >")
        print(" This method converts categorical variables into a form that could be provided to ML algorithms to do a better job in prediction:")
        print("  ✔ Converts each category value into a new column and assigns a 1 or 0 (notation for true/false).")
        print("  ✔ Ensures the data is ready for machine learning models without assuming any ordinal relationship.")
        print("  ✔ Helps to tackle the issue of 'curse of dimensionality' in a controlled manner.")
        print("✎ Note: Before encoding, ensure data uniformity and cleanliness.\n☻ Tip: Use `uniform_simple` or `uniform_smart` from `transform_cat()` for advanced categorical data cleaning.\n")

        # initialize dataframe to work with
        transformed_df = df.copy()

        # create an instance of one hot encoder
        onehot_encoder = OneHotEncoder(sparse_output=False)

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
        print("< ORDINAL ENCODING TRANSFORMATION >")
        print(" This method assigns an integer to each category value based on the provided ordinal order.")
        print("✎ Note: Ensure the provided ordinal map correctly reflects the desired order of categories for each variable.")
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
        print("< FREQUENCY ENCODING TRANSFORMATION >")
        print(" This method transforms categorical variables based on the frequency of each category.")
        print("✎ Note: Frequency encoding helps to retain the information about the category's prevalence.")
        print("☻ Tip: Useful for models where the frequency significance of categories impacts the prediction.\n")

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
        print("< TARGET ENCODING TRANSFORMATION* >")
        print(" This method encodes categorical variables based on the mean of the target variable for each category.")
        print("✎ Note: Target encoding captures the 'effect' of each category on the target variable.")
        print("☻ Tip: To prevent data leakage and overfitting, apply target encoding within a cross-validation loop, ensuring it's computed separately for each fold.\n")

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
        print("< BINARY ENCODING TRANSFORMATION >")
        print(" This method transforms categorical variables into binary columns, significantly reducing dimensionality for high cardinality features.")
        print("  ✔ Efficiently handles categories by representing them with binary codes.")
        print("  ✔ Reduces dataset size and model complexity compared to one-hot encoding.")
        print("✎ Note: Ideal for categorical variables with many unique categories.\n☻ Tip: For categories with limited unique values, consider if binary encoding aligns with your data strategy.\n")

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
