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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, mean_squared_error
from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.mixed_linear_model import MixedLM


def datetime_feature_extractor(df):
    feature_df = pd.DataFrame()
    for column in df.columns:
        # Ensure the column is in datetime format
        datetime_series = pd.to_datetime(df[column])
        # Extract datetime features
        feature_df[f"{column}_year"] = datetime_series.dt.year
        feature_df[f"{column}_month"] = datetime_series.dt.month
        feature_df[f"{column}_day"] = datetime_series.dt.day
    return feature_df


def data_preprocessing_core(
        df: pd.DataFrame,
        x_cols: list,
        y_col: str,
        data_state: str,
        test_size: float = 0.2,
        random_state: int = 42,
        numeric_imputer=SimpleImputer(strategy='median'),
        numeric_scaler=StandardScaler(),
        categorical_imputer=SimpleImputer(strategy='constant', fill_value='missing'),
        categorical_encoder=OneHotEncoder(handle_unknown='ignore'),
        text_vectorizer=CountVectorizer(),
        datetime_transformer=FunctionTransformer(datetime_feature_extractor, validate=False)
):
    """
    Performs comprehensive preprocessing on a dataset containing mixed data types.

    This function prepares a dataset for machine learning by handling numerical, categorical, text, and datetime data. It supports flexible imputation, scaling, encoding, and vectorization methods to cater to a wide range of preprocessing needs. The function automatically splits the data into training and test sets and applies the preprocessing steps defined by the user. It accommodates custom preprocessing steps for various data types, enhancing flexibility and control over the preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to preprocess.
    x_cols : list of str
        List of feature column names in `df` to include in the preprocessing.
    y_col : str
        The name of the target variable column in `df`.
    data_state : str
        Specifies the initial state of the data ('unprocessed' or 'preprocessed'). Default is 'unprocessed'.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Default is 0.2.
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split. Default is 42.
    numeric_imputer : sklearn imputer object, optional
        The imputation transformer for handling missing values in numerical data. Default is SimpleImputer(strategy='median').
    numeric_scaler : sklearn scaler object, optional
        The scaling transformer for numerical data. Default is StandardScaler().
    categorical_imputer : sklearn imputer object, optional
        The imputation transformer for handling missing values in categorical data. Default is SimpleImputer(strategy='constant', fill_value='missing').
    categorical_encoder : sklearn encoder object, optional
        The encoding transformer for categorical data. Default is OneHotEncoder(handle_unknown='ignore').
    text_vectorizer : sklearn vectorizer object, optional
        The vectorization transformer for text data. Default is CountVectorizer().
    datetime_transformer : callable, optional
        The transformation operation for datetime data. Default extracts year, month, and day as separate features.

    Returns
    -------
    x_train_processed : ndarray
        The preprocessed training feature set.
    x_test_processed : ndarray
        The preprocessed test feature set.
    y_train : Series
        The training target variable.
    y_test : Series
        The test target variable.
    task_type : str
        The type of machine learning task inferred from the target variable ('regression' or 'classification').

    Raises
    ------
    TypeError
        If input types are incorrect or if any transformer does not support the required methods.
    ValueError
        If specified columns are not found in `df`, or if `data_state` is not one of the expected values.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Age': np.random.randint(18, 35, size=100),
    ...     'Salary': np.random.normal(50000, 12000, size=100),
    ...     'Department': np.random.choice(['HR', 'Tech', 'Marketing'], size=100),
    ...     'Review': ['Good review']*50 + ['Bad review']*50,
    ...     'Employment Date': pd.date_range(start='2010-01-01', periods=100, freq='M')
    ... })
    >>> x_cols = ['Age', 'Salary', 'Department', 'Review', 'Employment Date']
    >>> y_col = 'Salary'
    >>> processed_data = data_preprocessing_core(df, x_cols, y_col, test_size=0.25, random_state=123)
    """

    # Error Handling
    # TypeErrors
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data_preprocessing_core(): The 'df' parameter must be a pandas DataFrame.")

    if not isinstance(x_cols, list) or not all(isinstance(col, str) for col in x_cols):
        raise TypeError("data_preprocessing_core(): The 'x_cols' parameter must be a list of strings representing column names.")

    if not isinstance(y_col, str):
        raise TypeError("data_preprocessing_core(): The 'y_col' parameter must be a string representing the target column name.")

    if not isinstance(data_state, str):
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

    if not hasattr(datetime_transformer, 'fit_transform'):
        raise TypeError("data_preprocessing_core(): The 'datetime_transformer' must be callable or support a 'transform' method.")

    # ValueErrors
    if data_state.lower() not in ['unprocessed', 'preprocessed'] or None:
        raise ValueError(f"data_preprocessing_core(): The data_state must be one of the following: \n- 'unprocessed': activates predict_ml() preprocessing capabilities.\n - 'preprocessed': user opts out of preprocessing (Warning: ensure your data is properly preprocessed for ML)")

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

        # define transformer names for reporting
        numeric_processor_name = type(numeric_imputer).__name__ + " & " + type(numeric_scaler).__name__
        categorical_processor_name = type(categorical_imputer).__name__ + " & " + type(categorical_encoder).__name__
        text_processor_name = type(text_vectorizer).__name__
        datetime_processor_name = "Custom DateTime Processing"

        # construct console output
        print(f"< PREPROCESSING DATA REPORT >\n ☻ Tip: You can customize the processors using appropriate parameters, please refer to documentation. \n")
        print(f"  ➡ Numerical features processed [using {numeric_processor_name}]: {', '.join(numeric_features) if numeric_features else 'None'}\n")
        print(f"  ➡ Categorical features processed [using {categorical_processor_name}]: {', '.join(categorical_features) if categorical_features else 'None'}\n")
        print(f"  ➡ Text features processed [using {text_processor_name}]: {', '.join(text_features) if text_features else 'None'}\n")
        print(f"  ➡ Datetime features processed [using {datetime_processor_name}]: {', '.join(datetime_features) if datetime_features else 'None'}\n")

        # define unprocessed features and output if any
        processed_features = set(numeric_features + categorical_features + text_features + datetime_features)
        all_features = set(x_cols)
        unprocessed_features = list(all_features - processed_features)
        if unprocessed_features:
            print(f"  ✘ Unprocessed features: {', '.join(unprocessed_features)}\n")
        else:
            pass

        return x_train_processed, x_test_processed, y_train, y_test, task_type
    else:
        # construct console output
        print(f"< PREPROCESSING DATA REPORT >\n  ➡ No preprocessing was done as the user indicated the data had already been preprocessed prior to using predict_ml().\n     [parameter: data_state = '{data_state}']\n")
        return x_train, x_test, y_train, y_test, task_type


# smoke test for data_preprocessing_core()
df = pd.DataFrame({
    'Age': np.random.randint(18, 35, size=100),
    'Salary': np.random.normal(50000, 12000, size=100),
    'Department': np.random.choice(['HR', 'Tech', 'Marketing'], size=100),
    'Review': ['Good review']*50 + ['Bad review']*50,
    'Employment Date': pd.date_range(start='2010-01-01', periods=100, freq='M')
})

# Define columns
x_cols = ['Age', 'Salary', 'Department', 'Review']
y_col = 'Salary'

# Preprocess the data
x_train_processed, x_test_processed, y_train, y_test, task_type = data_preprocessing_core(df, x_cols, y_col, data_state='unprocessed')





def model_recommendation_core(x_train, y_train, task_type: str, n_top_models=3):
    models_classification = {
        'LogisticRegression': LogisticRegression(max_iter=10000),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'SVC': SVC(probability=True),
        'KNeighborsClassifier': KNeighborsClassifier(),
    }

    models_regression = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'SVR': SVR(),
        'KNeighborsRegressor': KNeighborsRegressor(),
    }

    scoring_classification = {
        'Accuracy': 'accuracy',
        'Precision': 'precision_weighted',
        'Recall': 'recall_weighted',
        'F1': 'f1_weighted',
        'AUC': 'roc_auc',
    }

    scoring_regression = {
        'RMSE': 'neg_root_mean_squared_error',
        'MSE': 'neg_mean_squared_error',
        'MAE': 'neg_mean_absolute_error',
        'R2': 'r2',
    }

    if task_type == 'classification':
        models = models_classification
        scoring = scoring_classification
    elif task_type == 'regression':
        models = models_regression
        scoring = scoring_regression

    model_scores = {}
    for name, model in models.items():
        scores = cross_validate(model, x_train, y_train, cv=5, scoring=scoring)
        model_scores[name] = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring}

    # Example to display the scores
    for model, scores in model_scores.items():
        print(f"{model}:")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")
        print()

    # Further analysis for top model selection based on a primary metric can be added here
    # For simplicity, this example just prints the scores

    return model_scores


inference_models_continuousDV = {
    'OLS': OLS,
    # Add more models as needed
}

inference_models_categoricalDV = {
    'Logit': Logit,
    # MixedLM can be used for more complex cases with mixed effects.
    'MixedLM': MixedLM
}

model_scores = model_recommendation_core(x_train_processed, y_train, task_type)
