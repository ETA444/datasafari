"""

The idea is to have a multi-use ML tool that is not as absolute as some of the modules in this package
and especially not like predict_hypothesis() which basically auto-performs the whole hypo. testing flow for the user.

Here we want the user to have an advanced ML toolkit which provides value in multiple ways.

The idea for predict_ml() was born through what I now will call 'model explorer', this is the first
toolkit in this module. It basically goes through various models and based on various scores and criteria
tells the user which model is best. It gives the model back and the user can further work with it.

If the user chooses to user predict_ml further then they can plug in this model they got from the 'model explorer'
and it can be used for training which would lead to production and insights.

Alternatively the module can be used for inference, where through model explorer we get the best model
and similarly to predict_hypothesis(), where we will provide the user with a model summary, interpretation, tips and conclusions.

So in summary currently it seems the functionality will be:
- 'auto_recommender'
- 'auto_tuner'
- 'auto_inference'
(or better names lol we'll see)

"""
from datasafari.evaluator.evaluate_dtype import evaluate_dtype
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer


def data_preprocessing_core(
        df: pd.DataFrame,
        x_cols: list,
        y_col: str,
        data_state: str = 'unprocessed',
        test_size: float = 0.2,
        random_state: int = 42,
        numeric_imputer=SimpleImputer(strategy='median'),
        numeric_scaler=StandardScaler(),
        categorical_imputer=SimpleImputer(strategy='constant', fill_value='missing'),
        categorical_encoder=OneHotEncoder(handle_unknown='ignore'),
        text_vectorizer=CountVectorizer(),
        datetime_transformer=FunctionTransformer(lambda x: pd.to_datetime(x).apply(lambda x: [x.year, x.month, x.day]))
):
    """
    """
    # Error Handling
    # TypeErrors
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data_preprocessing_core(): The 'df' parameter must be a pandas DataFrame.")

    if not isinstance(x_cols, list) or not all(isinstance(col, str) for col in x_cols):
        raise TypeError("data_preprocessing_core(): The 'x_cols' parameter must be a list of strings representing column names.")

    if not isinstance(y_col, str):
        raise TypeError("data_preprocessing_core(): The 'y_col' parameter must be a string representing the target column name.")

    if not isinstance(data_state, str) or data_state.lower() not in ['unprocessed', 'preprocessed']:
        raise TypeError("data_preprocessing_core(): The 'data_state' parameter must be a string and one of ['unprocessed', 'preprocessed'].")

    if not isinstance(test_size, float) or not 0 < test_size < 1:
        raise TypeError("data_preprocessing_core(): The 'test_size' parameter must be a float between 0 and 1.")

    if not isinstance(random_state, int):
        raise TypeError("data_preprocessing_core(): The 'random_state' parameter must be an integer.")

    # Checking for the essential methods in imputers, scalers, encoders, and vectorizers
    if not (hasattr(numeric_imputer, 'fit_transform') or (hasattr(numeric_imputer, 'fit') and hasattr(numeric_imputer, 'transform'))):
        raise TypeError("data_preprocessing_core(): The 'numeric_imputer' must support 'fit_transform' or both 'fit' and 'transform' methods.")

    if not hasattr(numeric_scaler, 'fit_transform'):
        raise TypeError("data_preprocessing_core(): The 'numeric_scaler' must support 'fit_transform' method.")

    if not (hasattr(categorical_imputer, 'fit_transform') or (hasattr(categorical_imputer, 'fit') and hasattr(categorical_imputer, 'transform'))):
        raise TypeError("data_preprocessing_core(): The 'categorical_imputer' must support 'fit_transform' or both 'fit' and 'transform' methods.")

    if not hasattr(categorical_encoder, 'fit_transform'):
        raise TypeError("data_preprocessing_core(): The 'categorical_encoder' must support 'fit_transform' method.")

    if not hasattr(text_vectorizer, 'fit_transform'):
        raise TypeError("data_preprocessing_core(): The 'text_vectorizer' must support 'fit_transform' method.")

    if not callable(datetime_transformer):
        raise TypeError("data_preprocessing_core(): The 'datetime_transformer' must be callable or support a 'transform' method.")

    # ValueErrors
    if y_col not in df.columns:
        raise ValueError(f"data_preprocessing_core(): The specified target column '{y_col}' is not present in the DataFrame.")

    missing_cols = set(x_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"data_preprocessing_core(): The following feature columns are not present in the DataFrame: {', '.join(missing_cols)}")


    # Main Functionality
    # split data
    x_train, x_test, y_train, y_test = train_test_split(df[x_cols], df[y_col], test_size=test_size, random_state=random_state)

    # define task type based on the target variable data type
    y_dtype = evaluate_dtype(df, [y_col], output='dict')[y_col]
    task_type = 'regression' if y_dtype == 'numerical' else 'classification'

    if data_state.lower() == 'unprocessed':
        # evaluate data types to determine preprocessing needs
        data_types = evaluate_dtype(df, x_cols, output='dict')

        numeric_features = [col for col, dtype in data_types.items() if dtype == 'numerical']
        categorical_features = [col for col, dtype in data_types.items() if dtype == 'categorical']
        text_features = [col for col, dtype in data_types.items() if dtype == 'text']
        datetime_features = [col for col, dtype in data_types.items() if dtype == 'datetime']

        # preprocessing for numerical columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', numeric_imputer),
            ('scaler', numeric_scaler)
        ])

        # preprocessing for categorical columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', categorical_imputer),
            ('encoder', categorical_encoder)
        ])

        # preprocessing for text columns
        text_transformer = Pipeline(steps=[
            ('vectorizer', text_vectorizer)
        ])

        # combine preprocessing steps
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_features),
            ('datetime', datetime_transformer, datetime_features)
        ], remainder='passthrough')  # handle columns not specified in transformers

        # apply preprocessing
        x_train_processed = preprocessor.fit_transform(x_train)
        x_test_processed = preprocessor.transform(x_test)

        return x_train_processed, x_test_processed, y_train, y_test, task_type
    else:
        # if data is already preprocessed, simply return the splits and task type
        return x_train, x_test, y_train, y_test, task_type
