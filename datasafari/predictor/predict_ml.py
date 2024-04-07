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
from datasafari.transformer.transform_cat import transform_cat
from datasafari.transformer.transform_cat import transform_cat
from datasafari.evaluator.evaluate_dtype import evaluate_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from datasafari.evaluator import evaluate_dtype
import pandas as pd


def data_preprocessing_core(df: pd.DataFrame, x_cols: list, y_col: str, data_state: str = 'unprocessed', test_size: float = 0.2, random_state: int = 42, imputer=SimpleImputer(strategy='median'), scaler=StandardScaler(), categorical_na_fill_value: str = 'missing'):
    """
    """
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(df[x_cols], df[y_col], test_size=test_size, random_state=random_state)

    # Define task type used later in modeling
    y_dtype = evaluate_dtype(df, [y_col], output='dict')[y_col]
    task_type = 'regression' if y_dtype == 'numerical' else 'classification'

    if data_state.lower() == 'unprocessed':
        # Evaluate data types to determine preprocessing needs
        data_types = evaluate_dtype(df, x_cols, output='dict')

        numeric_features = [col for col, dtype in data_types.items() if dtype == 'numerical']
        categorical_features = [col for col, dtype in data_types.items() if dtype == 'categorical']

        # Define preprocessing for numerical columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', imputer),
            ('scaler', scaler)
        ])

        # Define preprocessing for categorical columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=categorical_na_fill_value)),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        # Apply preprocessing
        x_train_processed = preprocessor.fit_transform(x_train)
        x_test_processed = preprocessor.transform(x_test)

        return x_train_processed, x_test_processed, y_train, y_test, task_type
    else:
        # If data is already preprocessed, simply return the splits
        return x_train, x_test, y_train, y_test, task_type
