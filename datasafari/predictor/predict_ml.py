# used overall:
from typing import List, Dict, Any, Tuple, Callable, Union, Optional
from collections import defaultdict
from random import choice
import pandas as pd
import numpy as np
# mostly used within data_preprocessing_core():
from datasafari.evaluator.evaluate_dtype import evaluate_dtype
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, ParameterGrid
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
# mostly used within model_recommendation_core():
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# mostly used within the model_inference_core():
import statsmodels.formula.api as smf
# mostly used within the model_tuning_core():
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV


# Meta data #

# Available Classification Models
models_classification = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC(probability=True),
    'KNeighborsClassifier': KNeighborsClassifier(),
}

# Available Regression Models
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

# Available Classification Models (inference)
models_classification_inference = {
    'Logit': smf.logit,  # Logistic Regression
    'Probit': smf.probit,  # Probit Regression
    'MNLogit': smf.mnlogit,  # Multinomial Logistic Regression
    'Poisson': smf.poisson,  # Poisson Regression for count data
    'NegativeBinomial': smf.negativebinomial,  # Negative Binomial Regression for over-dispersed count data
    'GEE': smf.gee,  # Generalized Estimating Equations for repeated measurements
    'NominalGEE': smf.nominal_gee,  # GEE for nominal response
    'OrdinalGEE': smf.ordinal_gee  # GEE for ordinal response
}

# Available Regression Models (inference)
models_regression_inference = {
    'OLS': smf.ols,  # Ordinary Least Squares Linear Regression
    'WLS': smf.wls,  # Weighted Least Squares Linear Regression
    'GLS': smf.gls,  # Generalized Least Squares Linear Regression
    'RLM': smf.rlm,  # Robust Linear Models
    'QuantReg': smf.quantreg,  # Quantile Regression
    'GLSAR': smf.glsar,  # GLS with autoregressive errors
    'MixedLM': smf.mixedlm,  # Mixed Linear Model for hierarchical or longitudinal data
    'PHReg': smf.phreg  # Proportional Hazards Regression for survival analysis
}

# Available Scoring Metrics for Classification using Scikit
scoring_classification = {
    'Accuracy': 'accuracy',
    'Balanced Accuracy': 'balanced_accuracy',
    'Average Precision': 'average_precision',
    'Neg Brier Score': 'neg_brier_score',
    'F1 (Micro)': 'f1_micro',
    'F1 (Macro)': 'f1_macro',
    'F1 (Weighted)': 'f1_weighted',
    'Neg Log Loss': 'neg_log_loss',
    'Precision (Micro)': 'precision_micro',
    'Precision (Macro)': 'precision_macro',
    'Precision (Weighted)': 'precision_weighted',
    'Recall (Micro)': 'recall_micro',
    'Recall (Macro)': 'recall_macro',
    'Recall (Weighted)': 'recall_weighted',
    'Jaccard (Micro)': 'jaccard_micro',
    'Jaccard (Macro)': 'jaccard_macro',
    'Jaccard (Weighted)': 'jaccard_weighted',
    'ROC AUC (OVR)': 'roc_auc_ovr',
    'ROC AUC (OVO)': 'roc_auc_ovo',
}

# Available Scoring Metrics for Regression using Scikit
scoring_regression = {
    'EV': 'explained_variance',
    'MAE': 'neg_mean_absolute_error',
    'MSE': 'neg_mean_squared_error',
    'RMSE': 'neg_root_mean_squared_error',
    'MSLE': 'neg_mean_squared_log_error',
    'MedAE': 'neg_median_absolute_error',
    'R2': 'r2',
    'MPD': 'neg_mean_poisson_deviance',
    'MGD': 'neg_mean_gamma_deviance',
    'MAPE': 'neg_mean_absolute_percentage_error',
}

# Available Inference Metrics for Regression using Statsmodels
scoring_regression_inference = {
    'AIC': 'aic',
    'BIC': 'bic',
    'R-squared': 'rsquared',
    'Adjusted R-squared': 'rsquared_adj',
    'Log-Likelihood': 'llf'
}

# Available Inference Metrics for Classification using Statsmodels
scoring_classification_inference = {
    'AIC': 'aic',
    'BIC': 'bic',
    'Pseudo R-squared': 'prsquared',
    'Log-Likelihood': 'llf'
}

# Tips for Classification Scoring Metrics (if verbose > 1) - idea is to aid interpretation for non-advanced users
tips_scoring_classification = {
    'Accuracy': "Overall correctness, suitable for balanced classes. Higher scores indicate better performance.",
    'Balanced Accuracy': "Accuracy per class, great for imbalanced data. Higher values signal balanced class prediction capability.",
    'Average Precision': "Precision-recall balance, ideal for ranking tasks. Higher scores suggest better model precision.",
    'Neg Brier Score': "Probability calibration, lower is better, indicating accurate confidence in predictions.",
    'F1 (Micro)': "Aggregated F1 score, good for unbalanced data. High score shows effective overall class prediction.",
    'F1 (Macro)': "Mean F1 score across classes, for equal class emphasis. High values mean balanced performance across classes.",
    'F1 (Weighted)': "F1 score weighted by class, for imbalanced data. Reflects performance weighted towards prevalent classes.",
    'Neg Log Loss': "Model confidence, lower scores show better probability estimates.",
    'Precision (Micro)': "Overall model precision, useful in multiclass settings. High values indicate fewer false positives.",
    'Precision (Macro)': "Average precision, highlights class-specific performance. A high score denotes effective class differentiation.",
    'Precision (Weighted)': "Precision accounting for class imbalance. Focuses precision assessment on more frequent classes.",
    'Recall (Micro)': "Overall true positive rate. High scores show effectiveness in identifying positive instances.",
    'Recall (Macro)': "Balanced true positive rate, useful for equal class focus. Reflects consistent recall across classes.",
    'Recall (Weighted)': "Recall adjusted for class size, emphasizing larger classes. Indicates model's effectiveness on common classes.",
    'Jaccard (Micro)': "Intersection over union, measured globally. High values indicate broad prediction alignment with truth.",
    'Jaccard (Macro)': "Average IoU for each class, shows class-wise model agreement. High score signifies precise class predictions.",
    'Jaccard (Weighted)': "IoU weighted by class frequency. Targets performance improvement in dominant classes.",
    'ROC AUC (OVR)': "Area under ROC for multiclass, one-vs-rest. Higher scores mean better distinction between classes.",
    'ROC AUC (OVO)': "Area under ROC for multiclass, one-vs-one. Indicates model's discriminative power between any two classes.",
}

# Tips for Regression Scoring Metrics
tips_scoring_regression = {
    'EV': "Explains variance, perfect for models aiming high explanation power. Closer to 1 indicates better model.",
    'MaxError': "Worst-case error, critical for risk-sensitive models. Lower values denote reliability.",
    'MAE': "Average error magnitude, less sensitive to outliers. Lower MAE suggests higher precision.",
    'MSE': "Penalizes larger errors more, great for models where large errors are especially undesirable. Lower is better.",
    'RMSE': "Square root of MSE, on the target scale. Lower values indicate fewer and smaller errors.",
    'MSLE': "Focuses on relative errors, ideal for growth predictions. Lower scores reflect better accuracy on percentage scale.",
    'MedAE': "Middle error value, robust to outliers. Useful for skewed data, lower MedAE indicates central tendency accuracy.",
    'R2': "Proportion of variance explained, best for predictive models. Closer to 1, the more explanatory the model.",
    'MPD': "For count data, penalizing under/overestimations differently. Lower scores indicate Poisson conformity.",
    'MGD': "Assesses fit for gamma-distributed outcomes. Lower values show better adherence to gamma distribution.",
    'MAPE': "Percentage error, useful for comparative error measurement. Lower MAPE indicates better relative accuracy.",
}

# Available Tuners
tuners = {
    'grid': GridSearchCV,
    'random': RandomizedSearchCV,
    'bayesian': BayesSearchCV
}

# User-friendly tuner names for outputs
tuner_names = {
    'grid': 'GridSearchCV',
    'random': 'RandomizedSearchCV',
    'bayesian': 'BayesSearchCV'
}

# Default Parameter Grids for Classification Models
default_param_grids_classification = {
    'LogisticRegression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    },
    'DecisionTreeClassifier': {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300, 400],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2, 0.5],
        'max_depth': [3, 5, 7, 9]
    },
    'SVC': {
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
}

# Default Parameter Grids for Regression Models
default_param_grids_regression = {
    'LinearRegression': {
        # Linear Regression usually does not need hyperparameter tuning except for regularization
    },
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    },
    'Lasso': {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'selection': ['cyclic', 'random']
    },
    'DecisionTreeRegressor': {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'RandomForestRegressor': {
        'n_estimators': [100, 200, 300, 400],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoostingRegressor': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2, 0.5],
        'max_depth': [3, 5, 7, 9]
    },
    'SVR': {
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    },
    'KNeighborsRegressor': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
}


# Utility Functions (predict_ml() back-end) #
def datetime_feature_extractor(df: pd.DataFrame) -> pd.DataFrame:
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
        x_cols: List[str],
        y_col: str,
        data_state: str = 'unprocessed',
        test_size: float = 0.2,
        random_state: int = 42,
        numeric_imputer: TransformerMixin = SimpleImputer(strategy='median'),
        numeric_scaler: TransformerMixin = StandardScaler(),
        categorical_imputer: TransformerMixin = SimpleImputer(strategy='constant', fill_value='missing'),
        categorical_encoder: TransformerMixin = OneHotEncoder(handle_unknown='ignore'),
        text_vectorizer: TransformerMixin = CountVectorizer(),
        datetime_transformer: Callable[[pd.DataFrame], pd.DataFrame] = FunctionTransformer(datetime_feature_extractor, validate=False),
        verbose: int = 1
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, str]:
    """
    **Performs comprehensive preprocessing on a dataset containing mixed data types.**

    This function prepares a dataset for machine learning by handling numerical, categorical, text, and datetime data. It supports flexible imputation, scaling, encoding, and vectorization methods to cater to a wide range of preprocessing needs. The function automatically splits the data into training and test sets and applies the preprocessing steps defined by the user. It accommodates custom preprocessing steps for various data types, enhancing flexibility and control over the preprocessing pipeline.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to preprocess.

    x_cols : list of str
        List of feature column names in `df` to include in the preprocessing.

    y_col : str
        The name of the target variable column in `df`.

    data_state : str, optional, default: 'unprocessed'
        Specifies the initial state of the data ('unprocessed' or 'preprocessed').

    test_size : float, optional, default: 0.2
        Proportion of the dataset to include in the test split.

    random_state : int, optional, default: 42
        Controls the shuffling applied to the data before applying the split.

    numeric_imputer : sklearn imputer object, optional, default: SimpleImputer(strategy='median')
        The imputation transformer for handling missing values in numerical data.

    numeric_scaler : sklearn scaler object, optional, default: StandardScaler()
        The scaling transformer for numerical data.

    categorical_imputer : sklearn imputer object, optional, default: SimpleImputer(strategy='constant', fill_value='missing')
        The imputation transformer for handling missing values in categorical data.

    categorical_encoder : sklearn encoder object, optional, default: OneHotEncoder(handle_unknown='ignore')
        The encoding transformer for categorical data.

    text_vectorizer : sklearn vectorizer object, optional, default: CountVectorizer()
        The vectorization transformer for text data.

    datetime_transformer : callable, optional, default: None
        Transformer for datetime data.

            *Note: This parameter defaults to a custom datetime transformer, which extracts year, month, and day as separate features.*

    verbose : int, optional, default: 1
        The higher value the more output and information the user receives.


    Returns:
    --------
    x_train_processed : ndarray
        The preprocessed training feature set. *Note: The x_train will be untouched if ``data_state='preprocessed'``*

    x_test_processed : ndarray
        The preprocessed test feature set. *Note: The x_train will be untouched if ``data_state='preprocessed'``*

    y_train : Series
        The training target variable.

    y_test : Series
        The test target variable.

    task_type : str
        The type of machine learning task inferred from the target variable ('regression' or 'classification').


    Raises:
    -------
    TypeErrors:
        - If 'df' is not a pandas DataFrame.
        - If 'x_cols' is not a list of strings.
        - If 'y_col' is not a string.
        - If 'data_state' is not a string.
        - If 'test_size' is not a float.
        - If 'random_state' is not an integer.
        - If 'verbose' is not an integer
        - If numeric_imputer, numeric_scaler, categorical_imputer, categorical_encoder, text_vectorizer, or datetime_transformer do not support the required interface.
    ValueErrors:
        - If the `df` is empty.
        - If 'data_state' is not 'unprocessed' or 'preprocessed'.
        - If 'y_col' is not found in 'df'.
        - If specified 'x_cols' are not present in 'df'.
        - If 'test_size' is not between 0 and 1.
        - If 'df' does not contain enough data to split according to 'test_size'.


    Notes:
    ------
    The ``data_preprocessing_core()`` function is an integral part of the ``predict_ml()`` pipeline, designed to automate the preprocessing steps for machine learning tasks. It handles various data types, including numerical, categorical, text, and datetime, providing a streamlined process for preparing data for model training. This function uses Scikit-learn's transformers and custom functions to perform imputation, scaling, encoding, and vectorization, allowing users to customize these steps according to their needs.

    This function supports flexible preprocessing, accommodating custom transformations through its parameters. By specifying transformers for different data types, users can adapt the preprocessing to fit their dataset's specific characteristics. The function also splits the data into training and test sets, facilitating model evaluation and selection later in the pipeline.

    Designed with usability in mind, ``data_preprocessing_core()`` includes console output options controlled by 'verbose' parameter. This feature provides users with insights into the preprocessing steps taken (verbose > 0), including information on processed features and tips for further customization (verbose > 1).
    """
    if datetime_transformer is None:
        datetime_transformer = FunctionTransformer(datetime_feature_extractor, validate=False)

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

    if not isinstance(test_size, float):
        raise TypeError("data_preprocessing_core(): The 'test_size' parameter must be a float.")

    if not isinstance(random_state, int):
        raise TypeError("data_preprocessing_core(): The 'random_state' parameter must be an integer.")

    if not isinstance(verbose, int):
        raise TypeError("data_preprocessing_core(): 'verbose' must be an integer value.")

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
    if df.empty:
        raise ValueError("explore_num(): The input DataFrame is empty.")

    if not 0 < test_size < 1:
        raise ValueError("data_preprocessing_core(): The 'test_size' parameter must be a float between 0 and 1.")

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
        if verbose > 0:
            print(f"< PREPROCESSING DATA REPORT >")
            print(f" ☻ Tip: You can define your own SciKit preprocessors using the appropriate parameters, please refer to documentation. \n") if verbose > 1 else ''
            print(f"  ➡ Numerical features processed [using {numeric_processor_name}]: {', '.join(numeric_features) if numeric_features else 'None'}\n")
            print(f"  ➡ Categorical features processed [using {categorical_processor_name}]: {', '.join(categorical_features) if categorical_features else 'None'}\n")
            print(f"  ➡ Text features processed [using {text_processor_name}]: {', '.join(text_features) if text_features else 'None'}\n")
            print(f"  ➡ Datetime features processed [using {datetime_processor_name}]: {', '.join(datetime_features) if datetime_features else 'None'}\n")

        # define unprocessed features and output if any
        processed_features = set(numeric_features + categorical_features + text_features + datetime_features)
        all_features = set(x_cols)
        unprocessed_features = list(all_features - processed_features)
        if unprocessed_features:
            print(f"  ✘ Unprocessed features: {', '.join(unprocessed_features)}\n") if verbose > 0 else ''
        else:
            pass

        return x_train_processed, x_test_processed, y_train, y_test, task_type
    else:
        print(f"< PREPROCESSING DATA REPORT >\n  ➡ No preprocessing was done as the user indicated the data had already been preprocessed prior to using predict_ml().\n     [parameter: data_state = '{data_state}']\n") if verbose > 0 else ''
        return x_train, x_test, y_train, y_test, task_type


def calculate_composite_score(scores: dict, metric_weights: dict) -> float:
    """
    **Calculates a composite score based on individual metric scores and their respective weights.**

    This function aggregates multiple evaluation metrics into a single composite score by weighting each metric according to its importance, as defined in 'metric_weights'. A higher weight signifies greater importance of the metric towards the composite score. This approach allows for a balanced evaluation of model performance across various aspects.

    Parameters:
    ----------
    scores : dict
        A dictionary where keys are metric names (str) and values are their corresponding scores (float).
            - Example: ``{'Accuracy': 0.95, 'Precision': 0.90}``

    metric_weights : dict
        A dictionary where keys are metric names (str) and values are the weights (int or float) assigned to each metric.
            - Example: ``{'Accuracy': 5, 'Precision': 1}``

    Returns:
    -------
    float
        The composite score calculated as the weighted average of the provided metric scores.

    Raises:
    ------
    TypeError:
        - If 'scores' or 'metric_weights' is not a dictionary.
    ValueErrors:
        - If 'scores' or 'metric_weights' is empty.
        - If there are missing metric weights for any of the metrics provided in 'scores'.

    Examples:
    --------
    >>> scores = {'Accuracy': 0.95, 'Precision': 0.90}
    >>> metric_weights = {'Accuracy': 5, 'Precision': 1}
    >>> composite_score = calculate_composite_score(scores, metric_weights)
    >>> print(f"Composite Score: {composite_score:.2f}")

    Notes:
    ------
    - This function is utilized within the `model_recommendation_core()` of the `predict_ml()` pipeline, aimed at recommending the top `n` models based on a synthesized performance evaluation.
    - The composite score is derived by assigning weights to various scoring metrics, thereby enabling a prioritized and balanced assessment of model performance across multiple criteria.
    - Metrics for which lower values are traditionally better (e.g., RMSE) are transformed (either inverted or negated) prior to weight application, aligning all metrics to the "higher is better" principle for composite score calculation.
    - The calculation involves weighting each metric's score, summing these weighted scores, and normalizing the sum by the total weight, as detailed in the following formula:

        $$
        C = \frac{\sum_{m \in M} (w_m \cdot \text{adj}(s_m))}{\sum_{m \in M} w_m}
        $$

        where $\text{adj}(s_m)$ is the score adjustment function, ensuring a consistent interpretation across metrics, $w_m$ represents the weight of metric $m$, and $M$ is the set of all metrics.

    - The inversion or negation of scores for metrics where lower values are preferable ensures the composite score accurately reflects a model's overall efficacy, facilitating straightforward comparisons across diverse model configurations.
    - Review of metrics' adherence to the 'higher is better' framework indicates the systematic alignment of evaluation metrics, reinforcing the utility and interpretability of the composite scoring approach in model selection processes.
    """
    # Error Handling #

    # TypeErrors
    if not isinstance(scores, dict) or not isinstance(metric_weights, dict):
        raise TypeError("calculate_composite_score(): Both 'scores' and 'metric_weights' must be dictionaries.")

    # ValueErrors
    if not scores or not metric_weights:
        raise ValueError("calculate_composite_score(): 'scores' and 'metric_weights' dictionaries cannot be empty.")

    missing_metrics = set(scores.keys()) - set(metric_weights.keys())
    if missing_metrics:
        raise ValueError(f"calculate_composite_score(): Missing metric weights for: {', '.join(missing_metrics)}. Ensure 'metric_weights' includes all necessary metrics.")

    # Main Function #
    try:
        composite_score = sum(score * metric_weights.get(metric, 0) for metric, score in scores.items()) / sum(metric_weights.values())
    except Exception as e:
        raise ValueError(f"calculate_composite_score(): Error calculating composite score: {str(e)}")

    return composite_score


def model_recommendation_core(
        x_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        task_type: str,
        cv: int = 5,
        priority_metrics: List[str] = [],
        n_top_models: int = 3,
        verbose: int = 1
) -> Dict[str, Any]:
    """
    **Recommends top N machine learning models based on composite scores derived from multiple evaluation metrics.**

    This function is part of a broader machine learning pipeline, designed to facilitate model selection by automatically evaluating a range of models against a set of performance metrics, tailored to the specific needs of the analysis.

    Parameters:
    -----------
    x_train : Union[pd.DataFrame, np.ndarray]
        Training feature dataset.

    y_train : Union[pd.Series, np.ndarray]
        Training target variable.

    task_type : str
        Specifies the type of machine learning task: ``'classification'`` or ``'regression'``.

    priority_metrics : List[str], optional, default: []
        List of metric names given priority in model scoring.

    cv: int, optional, default: 5
        Determines the cross-validation splitting strategy.

    n_top_models : int, optional, default: 3
        Number of top models to recommend.

    verbose : int, optional, verbose: 1
        The higher value the more output and information the user receives.

    Returns:
    -------
    Dict[str, Any]
        Dictionary of top N recommended models, keyed by model name with model object as value.

    Raises:
    ------
    TypeErrors:
        - If 'x_train' is not a pandas DataFrame or NumPy ndarray.
        - If 'y_train' is not a pandas Series or NumPy ndarray.
        - If 'priority_metrics' is not a list.
        - If 'verbose' is not an integer.
    ValueErrors:
        - If 'task_type' is not 'classification' or 'regression'.
        - If 'n_top_models' is not an integer greater than 0.
        - If 'x_train' and 'y_train' do not have the same number of rows.
        - If 'x_train' or 'y_train' is empty.
        - If 'priority_metrics' contains duplicate values or items not representing metric names as strings.
        - If provided metric names in 'priority_metrics' are invalid or not supported, listing valid metric names for reference.
        - If provided metric names in 'priority_metrics' are not suitable for the 'task_type', listing valid metrics names for reference.
        - If 'n_top_models' exceeds the number of available models for the specified 'task_type'.


    Notes:
    -----
    The core leverages a composite score for model evaluation, which synthesizes scores across multiple metrics, weighted by the specified priorities. This method enables a holistic and nuanced model comparison, taking into account the multidimensional aspects of model performance.

        - **Priority Metrics:** Assigning weights (default: 5 for prioritized metrics, 1 for others) allows users to emphasize metrics they find most relevant, affecting the composite score calculation.

        - **Composite Score:** Calculated as a weighted average of metric scores, normalized by the total weight. This score serves as a basis for ranking models.

        - **Tips and Guidance:** Optional tips provide insights on interpreting and leveraging different metrics, enhancing informed decision-making in model selection.

        - **Ensuring 'Higher is Better' Across All Metrics:** For metrics where traditionally a lower score is better (e.g., RMSE), scores are transformed to align with the 'higher is better' principle used in composite score calculation. This transformation is inherent to the scoring configurations and does not require manual adjustment.
    """

    # Error handling #

    # TypeErrors
    if not isinstance(x_train, (pd.DataFrame, np.ndarray)):
        raise TypeError("model_recommendation_core(): 'x_train' must be a pandas DataFrame or NumPy ndarray.")

    if not isinstance(y_train, (pd.Series, np.ndarray)):
        raise TypeError("model_recommendation_core(): 'y_train' must be a pandas Series or NumPy ndarray.")

    if not isinstance(task_type, str) or task_type not in ['classification', 'regression']:
        raise ValueError("model_recommendation_core(): 'task_type' must be either 'classification' or 'regression'.")

    if not isinstance(priority_metrics, list):
        raise TypeError("model_recommendation_core(): 'priority_metrics' must be a list of scoring metric names.")

    if not isinstance(cv, int):
        raise TypeError("model_recommendation_core(): 'cv' must be an integer.")

    if not isinstance(n_top_models, int) or n_top_models <= 0:
        raise ValueError("model_recommendation_core(): 'n_top_models' must be an integer greater than 0.")

    if not isinstance(verbose, int):
        raise TypeError("model_recommendation_core(): 'verbose' must be an integer value.")

    # ValueErrors
    # ensure 'x_train' and 'y_train' are not empty
    if x_train.size == 0:
        raise ValueError("model_recommendation_core(): 'x_train' cannot be empty.")
    if y_train.size == 0:
        raise ValueError("model_recommendation_core(): 'y_train' cannot be empty.")

    # validate if x_train and y_train have compatible shapes
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("model_recommendation_core(): 'x_train' and 'y_train' must have the same number of rows.")

    # ensure 'priority_metrics' does not contain duplicates
    if len(priority_metrics) != len(set(priority_metrics)):
        raise ValueError("model_recommendation_core(): 'priority_metrics' should not contain duplicate values.")

    # ensure 'priority_metrics' list items are strings
    if not all(isinstance(metric, str) for metric in priority_metrics):
        raise ValueError("model_recommendation_core(): All items in 'priority_metrics' must be strings representing metric names.")

    # check if provided metrics are valid and remind users of valid options
    valid_metrics = set(scoring_classification.values()) | set(scoring_regression.values())
    invalid_metrics = [metric for metric in priority_metrics if metric not in valid_metrics]
    if invalid_metrics:
        valid_metric_list = ", ".join(sorted(valid_metrics))
        raise ValueError(f"model_recommendation_core(): Invalid metric(s) in 'priority_metrics': {', '.join(invalid_metrics)}.\n\nValid metrics are: {valid_metric_list}.")

    # check for valid metrics in relation to task type
    c_valid_metrics = set(scoring_classification.values())
    r_valid_metrics = set(scoring_regression.values())
    if task_type == 'classification':
        invalid_metrics = [metric for metric in priority_metrics if metric not in c_valid_metrics]
        if invalid_metrics:
            valid_metric_list = ", ".join(sorted(c_valid_metrics))
            raise ValueError(f"model_recommendation_core(): The following priority metrics are not valid for {task_type}: {', '.join(invalid_metrics)}.\n\nValid metrics for {task_type} are: {valid_metric_list}")
    elif task_type == 'regression':
        invalid_metrics = [metric for metric in priority_metrics if metric not in r_valid_metrics]
        if invalid_metrics:
            valid_metric_list = ", ".join(sorted(r_valid_metrics))
            raise ValueError(f"model_recommendation_core(): The following priority metrics are not valid for {task_type}: {', '.join(invalid_metrics)}.\n\nValid metrics for {task_type} are: {valid_metric_list}")

    # check if 'n_top_models' exceeds the number of available models for the task
    if task_type == 'classification' and n_top_models > len(models_classification):
        raise ValueError(f"model_recommendation_core(): 'n_top_models' cannot exceed the number of available classification models ({len(models_classification)}).")
    if task_type == 'regression' and n_top_models > len(models_regression):
        raise ValueError(f"model_recommendation_core(): 'n_top_models' cannot exceed the number of available regression models ({len(models_regression)}).")

    # Main Function #
    if task_type == 'classification':
        models = models_classification
        scoring = scoring_classification
        tips_scoring = tips_scoring_classification
    elif task_type == 'regression':
        models = models_regression
        scoring = scoring_regression
        tips_scoring = tips_scoring_regression

    # generate weights for priority metrics
    metric_weights = {metric_name: 5 if metric_func in priority_metrics else 1 for metric_name, metric_func in scoring.items()}

    model_scores = {}
    composite_scores = {}
    for name, model in models.items():
        scores = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring)

        average_scores = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring}
        composite_score = calculate_composite_score(average_scores, metric_weights)

        model_scores[name] = average_scores
        composite_scores[name] = composite_score

    top_models = sorted(composite_scores, key=composite_scores.get, reverse=True)[:n_top_models]

    if verbose > 0:
        print(f"< MODEL RECOMMENDATIONS >")
        print(f"The recommendation core has prioritized the following scoring metrics while choosing the best models: {', '.join([metric_name for metric_name, metric_func in scoring.items() if metric_func in priority_metrics])}\n") if priority_metrics else print(f"The recommendation core has not prioritized any metrics.\nTo prioritize a metric add it's name to the 'priority_metrics' list parameter. (e.g. priority_metrics=['explained_variance', 'neg_root_mean_squared_error']")
        [print(f" ☻ Tip on {scoring_metric}: {score_tip}\n") if scoring_metric in priority_metrics else '' for scoring_metric, score_tip in tips_scoring.items()] if verbose == 2 else ''
        [print(f" ☻ Tip on {scoring_metric}: {score_tip}\n") for scoring_metric, score_tip in tips_scoring.items()] if verbose == 3 else ''
        print(f"Best untuned models:")
        [print(f"  ➡ {model_name}()") for model_name in top_models]
        for model_name in top_models:
            print(f"\n► {model_name} (Composite Score: {composite_scores[model_name]:.4f}):")
            for metric, average_score in model_scores[model_name].items():
                print(f"  ⬥ {metric}: {average_score:.4f}")

    best_untuned_models = {model: models[model] for model in top_models}
    return best_untuned_models


def model_tuning_core(
        x_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        task_type: str,
        models: dict,
        priority_metrics: List[str] = None,
        refit_metric: str = None,
        priority_tuners: List[str] = None,
        custom_param_grids: dict = None,
        n_jobs: int = -1,
        cv: int = 5,
        n_iter_random: int = None,
        n_iter_bayesian: int = None,
        verbose: int = 1,
        random_state: int = 42
) -> Dict[str, Any]:
    """
    **Conducts hyperparameter tuning on a set of models using specified tuning methods and parameter grids, and returns the best tuned models along with their scores.**

    This function systematically applies grid search, random search, or Bayesian optimization to explore the hyperparameter space of given models. It supports customization of the tuning process through various parameters and outputs the best found configurations.

    Parameters:
    ----------
    x_train : Union[pd.DataFrame, np.ndarray]
        Training feature dataset.

    y_train : Union[pd.Series, np.ndarray]
        Training target variable.

    task_type : str
        Specifies the type of machine learning task: ``'classification'`` or ``'regression'``.

    models : dict
        Dictionary with model names as keys and model instances as values.

    priority_metrics : List[str], optional, default: None
        List of metric names given priority in model scoring.

    refit_metric : str, optional, default: None
        Metric to use for refitting the models in the machine learning pipeline. If ``None``, the function will use the first member of ``priority_matrics``. If no ``priority_metrics`` are provided, the function defaults to ``'Accuracy'`` for classification models and ``'MSE'`` for regression models.

    priority_tuners : List[str], optional, default: None
        List of tuner names to use for hyperparameter tuning. Valid tuners are ``'grid'``, ``'random'``, ``'bayesian'``.

    custom_param_grids : dict, optional, default: None
        Custom parameter grids to use, overriding the default grids if provided. Each entry should be a model name mapped to its corresponding parameter grid.

    n_jobs : int, optional, default: -1
        Number of jobs to run in parallel. -1 means using all processors/parallel processing.

    cv : int, optional, default: 5
        Number of cross-validation folds.

    n_iter_random : int, optional, default: 10
        Number of iterations for random search.

    n_iter_bayesian : int, optional, default: 50
        Number of iterations for Bayesian optimization.

    verbose : int, optional, default: 1
        Level of verbosity. The higher the number, the more detailed the console output.

    random_state : int, optional, default: 42
        Seed used by the random number generator.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the best models under each provided model name as keys. Values are dictionaries with keys: 'best_model' storing the model object of the best estimator and 'best_score' storing the corresponding score.

    Raises:
    ------
    TypeErrors:
        - If 'x_train' is not a pandas DataFrame or NumPy ndarray.
        - If 'y_train' is not a pandas Series or NumPy ndarray.
        - If 'task_type' is not a string.
        - If 'models' is not a dictionary with model names as keys and model instances as values.
        - If 'priority_metrics' is not None and is not a list of strings.
        - If 'priority_tuners' is not None and is not a list of strings.
        - If 'custom_param_grids' is not None and is not a dictionary.
        - If 'n_jobs', 'cv', 'n_iter_random', 'n_iter_bayesian', 'verbose', or 'random_state' are not integers.
        - If 'cv' is less than 1.
        - If 'n_iter_random' or 'n_iter_bayesian' is less than 1 when not None.
        - If 'refit_metric' is provided as a string but is not a callable or recognized metric name.
    ValueErrors:
        - If 'task_type' is not 'classification' or 'regression'.
        - If 'x_train' and 'y_train' do not have the same number of rows.
        - If 'x_train' or 'y_train' is empty (has zero elements).
        - If any element in 'priority_metrics' or 'priority_tuners' is not a string.
        - If 'priority_metrics' contains duplicate values.
        - If 'priority_tuners' contains unrecognized tuner names, not part of the expected tuners ('grid', 'random', 'bayesian').
        - If the specified 'refit_metric' is not applicable to the provided 'task_type' (e.g., using a regression metric for classification).
        - If 'n_iter_random_adjusted' or 'n_iter_bayesian_adjusted' becomes zero due to all combinations being previously tested, implying there are no new combinations to explore.
        - If 'n_iter_random' or 'n_iter_bayesian' is set to zero or a negative number.

    Notes:
    -----
        - **Integration with Tuning Methods**: This function utilizes scikit-learn's `GridSearchCV` and `RandomizedSearchCV`, along with scikit-optimize's `BayesSearchCV` for hyperparameter tuning. The choice of tuning method (`grid`, `random`, or `bayesian`) depends on the entries provided in the `priority_tuners` list.

        - **Skipping Repeated Combinations**: For Bayesian optimization (`BayesSearchCV`), the function is designed to skip evaluations of previously tested parameter combinations. This approach aims to enhance the efficiency and performance of the tuning process by reducing redundant computations.

        - **Parameter Grids Importance**: The quality and range of the parameter grids significantly influence the effectiveness of the tuning process. While default parameter grids are provided for convenience, it is recommended to supply customized parameter grids via `custom_param_grids` to ensure a thorough exploration of meaningful parameter combinations.

        - **Handling of User Warnings**: The `BayesSearchCV` from scikit-optimize occasionally emits warnings about the evaluation of repeated parameter points. This is a known issue within the library, unresolved for several years, which pertains to its internal handling of random state and the stochastic nature of the search algorithm. Although the function attempts to mitigate this by filtering out previously tested combinations, some warnings might still appear, especially under constraints of limited parameter grids or high `n_iter_bayesian` values.

        - **Refit Consideration**: The refit process, which re-trains the best estimator on the full dataset using the best-found parameters, is governed by the `refit_metric`. This metric should be carefully chosen to align with the overall objective and the specifics of the task at hand (classification or regression).

        - **Random State Usage**: The `random_state` parameter ensures reproducibility in the results of randomized search methods and Bayesian optimization, making the tuning outputs deterministic and easier to debug or review.

        - **Parallel Processing Capability**: Setting `n_jobs=-1` enables the function to use all available CPU cores for parallel processing, speeding up the computation especially beneficial when dealing with large datasets or complex models.
    """

    # Error Handling #

    # TypeErrors
    if not isinstance(x_train, (pd.DataFrame, np.ndarray)):
        raise TypeError("model_tuning_core(): 'x_train' must be a pandas DataFrame or NumPy ndarray.")

    if not isinstance(y_train, (pd.Series, np.ndarray)):
        raise TypeError("model_tuning_core(): 'y_train' must be a pandas Series or NumPy ndarray.")

    if not isinstance(task_type, str):
        raise TypeError("model_tuning_core(): 'task_type' must be a string.")

    if models is None or not isinstance(models, dict):
        raise TypeError("model_tuning_core(): 'models' must be a dictionary with model names as keys and model instances as values.")

    if priority_metrics is not None and not isinstance(priority_metrics, list):
        raise TypeError("model_tuning_core(): 'priority_metrics' must be a list of strings.")

    if priority_tuners is not None and not isinstance(priority_tuners, list):
        raise TypeError("model_tuning_core(): 'priority_tuners' must be a list of strings.")

    if custom_param_grids is not None and not isinstance(custom_param_grids, dict):
        raise TypeError("model_tuning_core(): 'custom_param_grids' must be a dictionary of parameter grids.")

    if not isinstance(n_jobs, int):
        raise TypeError("model_tuning_core(): 'n_jobs' must be an integer.")

    if not isinstance(cv, int):
        raise TypeError("model_tuning_core(): 'cv' must be an integer.")

    if n_iter_random is not None and (not isinstance(n_iter_random, int) or n_iter_random < 1):
        raise ValueError("model_tuning_core(): 'n_iter_random' must be a non-negative integer.")

    if n_iter_bayesian is not None and (not isinstance(n_iter_bayesian, int) or n_iter_bayesian < 1):
        raise ValueError("model_tuning_core(): 'n_iter_bayesian' must be a non-negative integer.")

    if not isinstance(verbose, int):
        raise TypeError("model_tuning_core(): 'verbose' must be an integer.")

    if not isinstance(random_state, int):
        raise TypeError("model_tuning_core(): 'random_state' must be an integer.")

    # ValueErrors
    if task_type not in ['classification', 'regression']:
        raise ValueError("model_tuning_core(): 'task_type' must be either 'classification' or 'regression'.")

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("model_tuning_core(): 'x_train' and 'y_train' must have the same number of rows.")

    if x_train.size == 0:
        raise ValueError("model_tuning_core(): 'x_train' cannot be empty.")

    if y_train.size == 0:
        raise ValueError("model_tuning_core(): 'y_train' cannot be empty.")

    if priority_metrics and any(not isinstance(metric, str) for metric in priority_metrics):
        raise ValueError("model_tuning_core(): All elements in 'priority_metrics' must be strings.")

    if priority_tuners and any(not isinstance(tuner, str) for tuner in priority_tuners):
        raise ValueError("model_tuning_core(): All elements in 'priority_tuners' must be strings representing the tuners' short names.")

    if priority_metrics and len(priority_metrics) != len(set(priority_metrics)):
        raise ValueError("model_tuning_core(): 'priority_metrics' contains duplicate entries.")

    if cv < 1:
        raise ValueError("model_tuning_core(): 'cv' must be an integer greater than 0.")

    # Check valid metrics based on task_type
    valid_metrics = set(scoring_classification.values()) if task_type == 'classification' else set(scoring_regression.values())
    invalid_metrics = set(priority_metrics) - valid_metrics if priority_metrics else set()

    if invalid_metrics:
        raise ValueError(f"model_tuning_core(): The following metrics are not valid for {task_type}: {', '.join(invalid_metrics)}. Valid metrics are: {', '.join(valid_metrics)}.")

    # Checking valid tuners
    valid_tuners = {'grid', 'random', 'bayesian'}  # Example of possible tuner names
    invalid_tuners = set(priority_tuners) - valid_tuners if priority_tuners else set()

    if invalid_tuners:
        raise ValueError(f"model_tuning_core(): The following tuners are not recognized: {', '.join(invalid_tuners)}. Valid tuners are: {', '.join(valid_tuners)}.")

    # Check refit_metric validity
    if isinstance(refit_metric, str) and refit_metric not in valid_metrics:
        raise ValueError(f"model_tuning_core(): 'refit_metric' {refit_metric} is not a valid metric. Choose from {', '.join(valid_metrics)}.")

    # Main Functionality #
    # initialize tracking for tested parameter combinations
    tested_combinations = defaultdict(set)

    # define default iterations if none provided
    n_iter_random = n_iter_random or 10
    n_iter_bayesian = n_iter_bayesian or 50

    scoring = scoring_classification if task_type == 'classification' else scoring_regression
    priority_scoring = {metric_name: metric_func for metric_name, metric_func in scoring.items() if metric_func in priority_metrics} if priority_metrics else scoring

    if refit_metric is None:
        if priority_metrics:
            first_priority_metric_name = [metric_name for metric_name, metric_func in scoring.items() if metric_func == priority_metrics[0]]
            refit_metric = first_priority_metric_name[0]
        else:
            refit_metric = 'Accuracy' if task_type == 'classification' else 'MSE'

    final_param_grids = default_param_grids_classification if task_type == 'classification' else default_param_grids_regression
    if custom_param_grids:
        final_param_grids.update(custom_param_grids)

    model_tuners = {tuner_name: tuners[tuner_name] for tuner_name in priority_tuners} if priority_tuners else tuners

    if verbose > 0:
        print("< TUNING INITIATED >")
        print(f"Starting model tuning for {task_type} modelling.\n")

        if priority_tuners:
            print(f"  ➡ Priority tuner(s): {', '.join([f'{tuner_names[tuner_short_name]}' for tuner_short_name in model_tuners.keys()])}")
        else:
            print(f"  ➡ Priority tuners: None (using default: {', '.join([f'{tuner_names[tuner_short_name]}' for tuner_short_name in model_tuners.keys()])}")

        if priority_metrics:
            print(f"  ➡ Priority metrics: {', '.join(priority_metrics)}")
        else:
            print("  ➡ Priority metrics: None (using default scoring criteria)")

        print(f"  ➡ Refit metric: {refit_metric}")

        if custom_param_grids:
            print(f"  ➡ Parameter grid: Custom (defined by user)")
        else:
            print("  ➡ Parameter grid: Default")

        if n_jobs == -1:
            print(f"  ➡ Parallel processing: ON")
        else:
            print(f"  ➡ Parallel processing: OFF (n_jobs = {n_jobs})")

        print(f"  ➡ Random state: {random_state}")

    tuned_models = {}
    for model_name, model_object in models.items():
        param_grid = final_param_grids.get(model_name, {})
        if not param_grid:
            if verbose > 0:
                print(f"\nNotice: Skipping tuning for {model_name} as no parameter grid is provided.")
            continue

        total_combinations = np.prod([len(v) for v in param_grid.values()])
        tested_fraction = len(tested_combinations[model_name]) / total_combinations if total_combinations > 0 else 0

        n_iter_random_adjusted = min(n_iter_random, int((1 - tested_fraction) * total_combinations))
        n_iter_bayesian_adjusted = min(n_iter_bayesian, int((1 - tested_fraction) * total_combinations))

        for tuner_short_name, tuner_class in model_tuners.items():

            if verbose > 0:
                print(f"\n► Tuning model: {model_name}")
                print(f"  ➡ Total possible param grid combinations: {total_combinations}\n") if verbose > 1 else ''

            if tuner_short_name == 'grid':
                tuner = tuner_class(
                    estimator=model_object, param_grid=param_grid, scoring=priority_scoring,
                    n_jobs=n_jobs, refit=refit_metric, cv=cv, verbose=verbose)
                if verbose > 0:
                    print(f"⬥ Running {tuner_names[tuner_short_name]} for {model_name}.")

            elif tuner_short_name == 'random':
                tuner = tuner_class(
                    estimator=model_object, param_distributions=param_grid, n_iter=n_iter_random_adjusted,
                    scoring=priority_scoring, n_jobs=n_jobs, refit=refit_metric, cv=cv, verbose=verbose,
                    random_state=random_state)
                if verbose > 0:
                    print(f"⬥ Running {tuner_names[tuner_short_name]} for {model_name}.")
                    if n_iter_random > n_iter_random_adjusted:
                        print(f"Notice: n_iter_random={n_iter_random} is reduced to max possible iterations {n_iter_random_adjusted} for {model_name}.\n")

            elif tuner_short_name == 'bayesian':
                if n_iter_bayesian_adjusted > 0:
                    tuner = tuner_class(
                        estimator=model_object, search_spaces=param_grid, n_iter=n_iter_bayesian_adjusted,
                        scoring=list(priority_scoring.values())[0], n_jobs=n_jobs, refit=refit_metric, cv=cv,
                        verbose=verbose, random_state=random_state)
                    if verbose > 0:
                        print(f"⬥ Running {tuner_names[tuner_short_name]} for {model_name}.")
                        if n_iter_bayesian > n_iter_bayesian_adjusted:
                            print(f"Notice: n_iter_bayesian={n_iter_bayesian} is reduced to max possible iterations {n_iter_bayesian_adjusted} for {model_name}.\n")

            else:
                if verbose > 1:
                    print(f"Notice: All parameter combinations for {model_name} have been tested. Skipping Bayesian optimization.")
                continue

            tuner.fit(x_train, y_train)
            best_model = tuner.best_estimator_
            best_score = tuner.best_score_

            if model_name not in tuned_models or best_score > tuned_models[model_name]['best_score']:
                tuned_models[model_name] = {'best_model': best_model, 'best_score': best_score}
                tested_combinations[model_name].add(tuple(tuner.best_params_.values()))

    if verbose > 0:
        print('\n< TUNING COMPLETED >')
        print("✔ Tuning summary:")
        for model, details in tuned_models.items():
            params_str = ', '.join(f"{key}={value}" for key, value in details['best_model'].get_params().items())
            print(f"  ➡ Best tuned version of {model} is {model}({params_str}) with score: {details['best_score']}")

    return tuned_models


def model_recommendation_core_inference(
        df: pd.DataFrame,
        formula: str,
        priority_models: List[str] = None,
        n_top_models: int = 3,
        model_kwargs: dict = None,
        verbose: int = 1
) -> Dict[str, Any]:
    """
    **Recommends top statistical models for inference based on user-specified preferences and formula.**

    This function evaluates various statistical models from statsmodels, each suitable for either regression or classification tasks determined dynamically by the nature of the target variable.


    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the data to fit the models.

    formula : str
        A patsy formula specifying the model. The independent variable is on the left of '~', while the dependent variables are on the right.

    priority_models : List[str], optional, default: None
        A list of model names to restrict the evaluation to specific models, otherwise all applicable models are evaluated.

    n_top_models : int, optional, default: 3
        Number of top-performing models to return based on sorted metrics.

    model_kwargs : dict, optional, default: None
        Dictionary mapping model names to dictionaries of additional keyword arguments to pass to the model constructors. This can be used to pass additional parameters required by specific models.

    verbose : int, optional, default: 1
        The verbosity level: 0 means silent, 1 outputs summary results, 2 includes detailed model summaries.


    Returns:
    -------
    Dict[str, Any]
        A dictionary with model names as keys and dictionaries as values. Each dictionary contains the 'model' object, 'metrics' dictionary with performance metrics, and potentially 'summary' if verbose > 1.


    Raises:
    ------
    TypeErrors:
        - If 'df' is not a pandas DataFrame, ensuring that the input data structure is correct for model fitting.
        - If 'formula' is not a string, verifying that the model formula is correctly specified as a string.
        - If 'priority_models' is provided and is not a list of strings, ensuring the user specifies a proper list of model names.
        - If 'model_kwargs' is provided and is not a dictionary, ensuring the correct format for passing additional keyword arguments to model constructors.
        - If 'verbose' is not an integer, verifying that the verbosity level is specified as an integer.
    ValueErrors:
        - If the input DataFrame is empty.
        - If 'formula' does not contain exactly one '~', which is necessary to separate the dependent and independent variables in the model specification.
        - If the specified target variable from 'formula' is not found in the DataFrame, ensuring the formula correctly references a column in the DataFrame.
        - If any variables specified in the 'formula' for independent variables are not found in the DataFrame, checking for the presence of all required variables in the DataFrame.
        - If 'n_top_models' is not a positive integer, ensuring that the number of models to return is specified correctly.


    Examples:
    --------
    >>> import datasafari
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'Age': np.random.randint(18, 70, size=100),
    ...     'Salary': np.random.normal(50000, 15000, size=100),
    ...     'Experience': np.random.randint(1, 30, size=100)
    ... })
    >>> formula = 'Salary ~ Age + Experience'
    >>> best_inference_models = model_recommendation_core_inference(
    ...     df,
    ...     formula,
    ...     verbose=2
    ... )
    >>> # Accessing the best model's object
    >>> best_model_name = list(best_inference_models.keys())[0]
    >>> best_model = best_inference_models[best_model_name]['model']
    >>> # Viewing the summary of the best model
    >>> print(best_model.summary())
    >>> # Extracting AIC of the best model
    >>> best_model_aic = best_inference_models[best_model_name]['metrics']['AIC']
    >>> print(f"The best model according to AIC is {best_model_name} with an AIC of {best_model_aic:.2f}")

    Notes:
    -----
    - **Dynamic Model Evaluation**: Depending on the datatype of the target variable specified in the formula, the function dynamically decides whether to treat the problem as a regression or classification task, using appropriate metrics and models for each.

    - **Handling Model Specific Requirements**: This function allows passing custom arguments to model constructors to handle models that require specific parameters via `model_kwargs`.

    - **Metric Adjustments**: For metrics where a lower value is better (e.g., AIC, BIC), these are adjusted to be compared directly alongside higher-is-better metrics like R-squared, by negating their values during sorting.

    - **Verbose Output**: The function provides different levels of output detail which can help in diagnosing model fit or understanding model performance.

    - **Error Handling**: The function will report and skip models that encounter errors during fitting, allowing for robust execution even if some models are not applicable to the provided data or formula.
    """

    # Error handling #
    # TypeErrors
    if not isinstance(df, pd.DataFrame):
        raise TypeError("model_recommendation_core_inference(): 'df' must be a pandas DataFrame.")

    if not isinstance(formula, str):
        raise TypeError("model_recommendation_core_inference(): 'formula' must be a string.")

    if priority_models is not None and not isinstance(priority_models, list):
        raise TypeError("model_recommendation_core_inference(): 'priority_models' must be a list of strings.")

    if not isinstance(n_top_models, int) or n_top_models < 1:
        raise ValueError("model_recommendation_core_inference(): 'n_top_models' must be an integer greater than 0.")

    if model_kwargs is not None and not isinstance(model_kwargs, dict):
        raise TypeError("model_recommendation_core_inference(): 'model_kwargs' must be a dictionary.")

    if not isinstance(verbose, int):
        raise TypeError("model_recommendation_core_inference(): 'verbose' must be an integer.")

    # ValueErrors
    # Ensure non-empty DataFrame
    if df.empty:
        raise ValueError("model_recommendation_core_inference(): The input DataFrame is empty.")

    if formula.count('~') != 1:
        raise ValueError("model_recommendation_core_inference(): 'formula' must include exactly one '~' to separate dependent and independent variables.")

    y_col = formula.split('~')[0].strip()
    if y_col not in df.columns:
        raise ValueError(f"model_recommendation_core_inference(): Specified target variable '{y_col}' is not in DataFrame.")

    # Check for presence of all variables specified in the formula in the DataFrame
    independent_vars = formula.split('~')[1]
    missing_vars = [var.strip() for var in independent_vars.replace('+', ' ').split() if var.strip() not in df.columns]
    if missing_vars:
        raise ValueError(f"model_recommendation_core_inference(): The following independent variables are not in DataFrame: {', '.join(missing_vars)}.")

    # Main Functionality #
    # define task type based on the target variable data type
    y_dtype = evaluate_dtype(df, [y_col], output='dict')[y_col]
    task_type = 'regression' if y_dtype == 'numerical' else 'classification'

    # define models to choose from based on task type
    models = models_classification_inference if task_type == 'classification' else models_regression_inference
    scoring = scoring_classification_inference if task_type == 'classification' else scoring_regression_inference

    # consider priority models if any
    models = {model_name: model_func for model_name, model_func in models.items() if (priority_models and model_name in priority_models) or not priority_models}

    model_results = {}
    for name, model_func in models.items():
        try:
            # use provided kwargs if applicable for the model
            kwargs = model_kwargs.get(name, {}) if model_kwargs is not None else None
            model = model_func(formula, df, **kwargs).fit() if kwargs else model_func(formula, df).fit()
            metrics = {}
            for metric, attr in scoring.items():
                try:
                    metrics[metric] = getattr(model, attr)
                except (NotImplementedError, AttributeError):
                    if verbose > 1:
                        print(f"Warning: {metric} is not supported by the model {name}.")

            if metrics:
                model_results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'adjusted_metrics': {
                        metric: (-value if metric in ['AIC', 'BIC'] else value)
                        for metric, value in metrics.items()
                    }
                }
        except Exception as e:
            if verbose > 0:
                print(f"Warning: Error fitting model {name}: {str(e)}")
            continue

    # sorting models based on adjusted metrics
    sorted_models = sorted(
        [(name, details) for name, details in model_results.items() if not all(pd.isna(value) for value in details['metrics'].values())],
        key=lambda x: tuple(x[1]['adjusted_metrics'][metric] for metric in scoring.keys() if metric in x[1]['adjusted_metrics']),
        reverse=True
    )[:n_top_models]

    if verbose > 0:
        print(f"\n< MODEL RECOMMENDATIONS >")
        print(f" ☻ Tip: Use verbose = 2 to see model summaries, or access the models directly in the returned dictionary object.") if verbose < 2 else ''
        for name, details in sorted_models:
            print(f"\n\n ➡ Model: {name}")
            for metric, value in details['metrics'].items():
                print(f"  ⬥ {metric}: {value:.4f}")
            if verbose > 1 and 'model' in details:
                print(details['model'].summary())

    return {name: details for name, details in sorted_models}


# Main Function (front-end) #
def predict_ml(
        df: pd.DataFrame,
        x_cols: List[str] = None,
        y_col: str = None,
        formula: str = None,
        data_state: str = 'unprocessed',
        n_top_models: int = 3,
        test_size: float = 0.2,
        cv: int = 5,
        random_state: int = 42,
        priority_metrics: List[str] = [],
        refit_metric: str = None,
        priority_tuners: List[str] = None,
        custom_param_grids: dict = None,
        n_jobs: int = -1,
        n_iter_random: int = None,
        n_iter_bayesian: int = None,
        priority_models: List[str] = None,
        model_kwargs: dict = None,
        verbose: int = 1,
        numeric_imputer: TransformerMixin = SimpleImputer(strategy='median'),
        numeric_scaler: TransformerMixin = StandardScaler(),
        categorical_imputer: TransformerMixin = SimpleImputer(strategy='constant', fill_value='missing'),
        categorical_encoder: TransformerMixin = OneHotEncoder(handle_unknown='ignore'),
        text_vectorizer: TransformerMixin = CountVectorizer(),
        datetime_transformer: Callable[[pd.DataFrame], pd.DataFrame] = None
) -> Dict[str, Any]:
    r"""

    **Streamline the entire process of data preprocessing, model selection, and tuning, delivering optimal model recommendations based on the data provided.**

    Depending on the inputs, this function can either perform statistical inference or predictive model selection using machine learning.
        - **Machine Learning Pipeline**: Focuses on predictive model selection and hyperparameter tuning using scikit-learn. It includes preprocessing (optional), model recommendation based on specified metrics, and tuning using grid search, random search, or Bayesian optimization.
        - **Inference Pipeline**: Utilizes statsmodels for detailed statistical analysis and model fitting based on a specified formula. This pipeline is tailored for users seeking statistical inference, providing metrics such as AIC, BIC, and R-squared. This pipeline assumes the data to have been preprocessed appropriately beforehand.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the dataset to be analyzed.

    x_cols : List[str], optional, default: None
        List of column names to be used as features for machine learning model recommendation.

    y_col : str, optional, default: None
        Column name to be used as the target for machine learning model recommendation.

    formula : str, optional, default: None
        A Patsy formula for specifying the model in the case of statistical inference.

    data_state : str, optional, default: 'unprocessed'
        Specifies the initial state of the data (``'unprocessed'`` or ``'preprocessed'``).

        - ``'unprocessed'`` will trigger the customizable preprocessing procedure.
        - ``'preprocessed'`` will omit the preprocessing procedure. *Only suitable for preprocessed data!*

    n_top_models : int, optional, default: 3
        Number of top models to recommend from the evaluation.

    test_size : float, optional, default: 0.2
        Proportion of the dataset to be used as the test set.

    cv : int, optional, default: 5
        Number of cross-validation folds.

    random_state : int, optional, default: 42
        Controls the shuffling applied to the data before applying the split.

    priority_metrics : List[str], optional, default: []
        Metrics to prioritize in model evaluation in the machine learning pipeline. *Note: The list members must be in the correct format as specified below.*

            **Available Metrics:**

            - **Regression:** ``'explained_variance'``, ``'neg_mean_absolute_error'``, ``'neg_mean_squared_error'``, ``'neg_root_mean_squared_error'``, ``'neg_mean_squared_log_error'``, ``'neg_median_absolute_error'``, ``'r2'``, ``'neg_mean_poisson_deviance'``, ``'neg_mean_gamma_deviance'``, ``'neg_mean_absolute_percentage_error'``
            - **Classification:** ``'accuracy'``, ``'balanced_accuracy'``, ``'average_precision'``, ``'neg_brier_score'``, ``'f1_micro'``, ``'f1_macro'``, ``'f1_weighted'``, ``'neg_log_loss'``, ``'precision_micro'``, ``'precision_macro'``, ``'precision_weighted'``, ``'recall_micro'``, ``'recall_macro'``, ``'recall_weighted'``, ``'jaccard_micro'``, ``'jaccard_macro'``, ``'jaccard_weighted'``, ``'roc_auc_ovr'``, ``'roc_auc_ovo'``

    refit_metric : str, optional, default: None
        Metric to use for refitting the models in the machine learning pipeline. *Note: The string must be in the correct format as specified below.*

        - If ``None``, the function will use the first member of ``priority_matrics``.
        - If ``None`` and no ``priority_metrics`` are provided, the function defaults to ``'Accuracy'`` for classification models and ``'MSE'`` for regression models.

            **Available Refit Metrics:**

            - **Regression:** ``'EV'``, ``'MAE'``, ``'MSE'``, ``'RMSE'``, ``'MSLE'``, ``'MedAE'``, ``'R2'``, ``'MPD'``, ``'MGD'``, ``'MAPE'``
            - **Classification:** ``'Accuracy'``, ``'Balanced Accuracy'``, ``'Average Precision'``, ``'Neg Brier Score'``, ``'F1 (Micro)'``, ``'F1 (Macro)'``, ``'F1 (Weighted)'``, ``'Neg Log Loss'``, ``'Precision (Micro)'``, ``'Precision (Macro)'``, ``'Precision (Weighted)'``, ``'Recall (Micro)'``, ``'Recall (Macro)'``, ``'Recall (Weighted)'``, ``'Jaccard (Micro)'``, ``'Jaccard (Macro)'``, ``'Jaccard (Weighted)'``, ``'ROC AUC (OVR)'``, ``'ROC AUC (OVO)'``

    priority_tuners : List[str], optional, default: None
        Tuners to use for hyperparameter tuning in the machine learning pipeline. *Note: The list members must be in the correct format as specified below.*

            **Available Tuners:** ``'grid'``, ``'random'``, ``'bayesian'``

    custom_param_grids : dict, optional, default: None
        Custom parameter grids for tuning in the machine learning pipeline. *Note: Template dictionaries are provided at the end of this page.*

    n_jobs : int, optional, default: -1
        Number of jobs to run in parallel. -1 means using all processors/parallel processing.

    n_iter_random : int, optional, default: None
        Number of iterations for random search tuning in the machine learning pipeline.

    n_iter_bayesian : int, optional, default: None
        Number of iterations for Bayesian optimization in the machine learning pipeline.

    priority_models : List[str], optional, default: None
        Specific models to evaluate in the inference pipeline. *Note: The list members must be in the correct format as specified below.*

        - If ``None`` the function will assess all appropriate models.

            **Available Inferential Models:**

            - **Regression:** ``'OLS'``, ``'WLS'``, ``'GLS'``, ``'RLM'``, ``'QuantReg'``, ``'GLSAR'``, ``'MixedLM'``, ``'PHReg'``
            - **Classification:** ``'Logit'``, ``'Probit'``, ``'MNLogit'``, ``'Poisson'``, ``'NegativeBinomial'``, ``'GEE'``, ``'NominalGEE'``, ``'OrdinalGEE'``

    model_kwargs : dict, optional, default: None
        Keyword arguments to pass to model constructors in the inference pipeline.

    verbose : int, optional, default: 1
        Level of verbosity in output.

    numeric_imputer : TransformerMixin, optional, default: SimpleImputer(strategy='median')
        Imputer for handling missing values in numerical data, if ``data_state='unprocessed'``.

            Any imputer from ``scikit.impute`` can be used instead of the default.

    numeric_scaler : TransformerMixin, optional, default: StandardScaler()
        Scaler for numerical data, if ``data_state='unprocessed'``.

           Any scaler from ``scikit.preprocessing`` can be used instead of the default.

    categorical_imputer : TransformerMixin, optional, default: SimpleImputer(strategy='constant', fill_value='missing')
        Imputer for handling missing values in categorical data, if ``data_state='unprocessed'``.

           Any imputer from ``scikit.preprocessing`` can be used instead of the default.

    categorical_encoder : TransformerMixin, optional, default: OneHotEncoder(handle_unknown='ignore')
        Encoder for categorical data, if ``data_state='unprocessed'``.

            Any enoder from ``scikit.preprocessing`` can be used instead of the default.

    text_vectorizer : TransformerMixin, optional, default: CountVectorizer()
        Vectorizer for text data, if ``data_state='unprocessed'``.

            Any vectorizer from ``sklearn.feature_extraction.text`` can be used instead of the default.

    datetime_transformer : callable, optional, default: None
        Transformer for datetime data, if ``data_state='unprocessed'``.

            *Note: This parameter defaults to a custom datetime transformer, which extracts year, month, and day as separate features. This is an experimental feature and it is not recommended to use other solutions.*


    Returns:
    --------
    Dict[str, Any]
        Depending on the operation mode, the dictionary contains either:
            - top machine learning models and their evaluation metrics,
            - top statistical models along with their fit statistics.


    Examples:
    ---------
    **Import necessary libraries and generate a DataFrame for examples:**

    >>> import datasafari
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'Age': np.random.randint(18, 35, size=100),
    ...     'Salary': np.random.normal(50000, 12000, size=100),
    ...     'Department': np.random.choice(['HR', 'Tech', 'Marketing'], size=100),
    ...     'Review': ['Good review']*50 + ['Bad review']*50,
    ...     'Employment Date': pd.date_range(start='2010-01-01', periods=100, freq='M')
    ... })

    Machine Learning Pipeline Examples
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **Simple Machine Learning Pipeline:**

    >>> x_cols = ['Age', 'Salary', 'Department', 'Review', 'Employment Date']
    >>> y_col = 'Salary'
    >>> ml_models = predict_ml(df, x_cols=x_cols, y_col=y_col, verbose=2)

    **Utilizing Priority Metrics and Refit Metric in Machine Learning Pipeline:**

    >>> priority_metrics = ['neg_mean_squared_error', 'r2']
    >>> ml_models_priority_metrics = predict_ml(
    ...     df,
    ...     x_cols=x_cols,
    ...     y_col=y_col,
    ...     priority_metrics=priority_metrics,
    ...     refit_metric='r2',
    ...     verbose=2
    ... )

    **Integrating Priority Tuners with Custom Parameter Grids:**

    >>> custom_grids = {
    ...     'RandomForestClassifier': {
    ...         'n_estimators': [100, 200],
    ...         'max_depth': [None, 10, 20]
    ...     }
    ... }
    >>> priority_tuners = ['random', 'grid']
    >>> ml_models_with_custom_tuning = predict_ml(
    ...     df,
    ...     x_cols=x_cols,
    ...     y_col=y_col,
    ...     priority_metrics=priority_metrics,
    ...     refit_metric='r2',
    ...     priority_tuners=priority_tuners,
    ...     custom_param_grids=custom_grids,
    ...     verbose=2
    ... )

    **Advanced Machine Learning Pipeline Using Bayesian Optimization:**

    >>> priority_tuners = ['bayesian']
    >>> n_iter_bayesian = 50
    >>> ml_models_bayesian = predict_ml(
    ...     df,
    ...     x_cols=x_cols,
    ...     y_col=y_col,
    ...     priority_metrics=priority_metrics,
    ...     refit_metric='r2',
    ...     priority_tuners=priority_tuners,
    ...     n_iter_bayesian=n_iter_bayesian,
    ...     verbose=2
    ... )


    Inference Pipeline Examples
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    **Simple Inference Example Using a Statistical Model:**

    Find the relationship between `Salary` and `Age` using Ordinary Least Squares (OLS) regression:

    >>> formula = 'Salary ~ Age'
    >>> inference_result_ols = predict_ml(df, formula=formula, verbose=2)

    **Using a Categorical Predictor with OLS:**

    Incorporate a categorical variable (`Department`) in the model to examine its effect on `Salary`:

    >>> formula = 'Salary ~ Age + C(Department)'
    >>> inference_result_ols_categorical = predict_ml(df, formula=formula, verbose=2)

    **Advanced Inference with Multiple Models and Specific Metrics:**

    Compare multiple regression models focusing on their fit statistics:

    >>> priority_models = ['OLS', 'WLS', 'GLS']
    >>> advanced_inference_models = predict_ml(
    ...     df,
    ...     formula=formula,
    ...     priority_models=priority_models,
    ...     verbose=2
    ... )

    **Inference with Robust Regression Model:**

    Utilize Robust Linear Models (RLM) to mitigate the influence of outliers:

    >>> formula = 'Salary ~ Age + C(Department)'
    >>> robust_inference_result = predict_ml(
    ...     df,
    ...     formula=formula,
    ...     priority_models=['RLM'],
    ...     verbose=2
    ... )

    **Mixed Linear Model for Hierarchical or Longitudinal Data:**

    Apply a Mixed Linear Model (MixedLM) if the data structure involves nested or grouped observations:

    >>> formula = 'Salary ~ Age + C(Department) + (1|Employment Date)'
    >>> mixedlm_inference_result = predict_ml(
    ...     df,
    ...     formula=formula,
    ...     priority_models=['MixedLM'],
    ...     verbose=2
    ... )


    Notes:
    ------

    Pipelines Explained
    ^^^^^^^^^^^^^^^^^^^

    Machine Learning Pipeline
    +++++++++++++++++++++++++
        1. **Data Preprocessing (optional):** optionally prepares a dataset for machine learning by handling numerical, categorical, text, and datetime data.
            - It supports flexible imputation, scaling, encoding, and vectorization methods to cater to a wide range of preprocessing needs.
            - The function automatically splits the data into training and test sets and applies the preprocessing steps defined by the user.
            - It accommodates custom preprocessing steps for various data types, enhancing flexibility and control over the preprocessing pipeline.

        2. **Evaluation of Untuned Models:** leverages a composite score for model evaluation, which synthesizes scores across multiple metrics, weighted by the specified priorities. This method enables a holistic and nuanced model comparison, taking into account the multidimensional aspects of model performance.
            - **Priority Metrics:** Assigning weights (default: 5 for prioritized metrics, 1 for others) allows users to emphasize metrics they find most relevant, affecting the composite score calculation.
            - **Composite Score:** Calculated as a weighted average of metric scores, normalized by the total weight. This score serves as a basis for ranking models. The formula for the composite score is given by:

            .. math::

                C = \frac{\sum_{m \in M} (w_m \cdot \text{adj}(s_m))}{\sum_{m \in M} w_m}

            Where:
                - :math:`\text{adj}(s_m)` is the score adjustment function, ensuring a consistent interpretation across metrics. Metrics for which lower values are traditionally better (e.g., RMSE, MAE) are inverted or negated prior to weight application, aligning all metrics to the "higher is better" principle for score calculation.
                - :math:`w_m` represents the weight of metric :math:`m`.
                - :math:`M` is the set of all metrics considered in the evaluation.

        3. **Model Tuning:** Uses top N untuned models to tune. Systematically applies grid search, random search, or Bayesian optimization to explore the hyperparameter space of given models. It supports customization of the tuning process through various parameters and outputs the best found configurations.


    Statistical Inference Pipeline
    ++++++++++++++++++++++++++++++
        1. **Determination of Task Type:** First, the function identifies whether the analysis involves regression or classification. This categorization is based on the datatype of the target variable specified in the formula:
            - **Regression:** Applied if the target variable is numerical.
            - **Classification:** Applied if the target variable is categorical.
            This step ensures that the appropriate statistical models and metrics are selected for the analysis.

        2. **Model Selection:** Based on the task type determined in the previous step, the function selects from a pre-defined set of models suitable for either regression or classification:
            - Models and their respective functions are predefined in the `models_classification_inference` or `models_regression_inference` dictionaries, depending on whether the task is classification or regression.
            - The user has the option to limit the evaluation to a subset of models through the `priority_models` parameter, enhancing focus and computational efficiency.

        3. **Model Evaluation:** Each selected model is fitted to the data using the formula provided:
            - The function iterates over each model, passing any user-defined keyword arguments specific to that model using the `model_kwargs` dictionary. This allows for customized model configurations.
            - Models are fitted using their respective statistical functions from the statsmodels API, adhering to the specifications in the formula.

        4. **Metrics Calculation:** After fitting, the function evaluates each model using a set of predefined metrics appropriate for the task type:
            - Certain metrics, particularly those for which a lower value indicates better performance (e.g., AIC, BIC), are adjusted to fit a common scoring scheme where higher values indicate better model performance.

        5. **Model Ranking and Output:** Finally, the models are ranked based on their performance metrics:
            - A sorted list of models is generated based on the adjusted metrics, allowing the top-performing models to be identified.
            - The function returns the top `n_top_models` as specified, including their fitted model objects and performance metrics, facilitating further analysis or validation by the user.
            - If verbose output is enabled, the function provides detailed summaries of the top models, aiding in interpretive and diagnostic processes.




    Available Metadata
    ^^^^^^^^^^^^^^^^^^^

    Below you can find all of the models, scoring metrics and tuners ``predict_ml()`` is equipped with. We also provide default parameter grids for users who do not wish to provide it.

    ML-oriented Models
    ++++++++++++++++++

        **Classification Models**
            - **LogisticRegression**: Provides logistic regression for binary classification.
            - **DecisionTreeClassifier**: Offers decision tree algorithms for classification.
            - **RandomForestClassifier**: Implements a random forest for classification.
            - **GradientBoostingClassifier**: Applies gradient boosting techniques for classification.
            - **SVC**: Support Vector Classifier with enabled probability estimates.
            - **KNeighborsClassifier**: Utilizes k-nearest neighbors voting classification.

        **Regression Models**
            - **LinearRegression**: Ordinary least squares Linear Regression.
            - **Ridge**: Ridge regression with L2 regularization.
            - **Lasso**: Lasso regression with L1 regularization.
            - **DecisionTreeRegressor**: Regression based on decision trees.
            - **RandomForestRegressor**: Random forest algorithm for regression.
            - **GradientBoostingRegressor**: Gradient boosting for regression.
            - **SVR**: Epsilon-Support Vector Regression.
            - **KNeighborsRegressor**: Regression based on k-nearest neighbors.

    Inference-oriented Models
    +++++++++++++++++++++++++

    These models are specifically used for statistical inference, allowing for detailed statistical analysis.

        **Classification Inference Models**
            - **Logit**: Logistic regression for binary classification.
            - **Probit**: Probit model for binary classification.
            - **MNLogit**: Multinomial logistic regression for handling multiple categories.
            - **Poisson**: Poisson model for count data.
            - **NegativeBinomial**: Negative binomial model for count data with over-dispersion.
            - **GEE**: Generalized Estimating Equations for longitudinal data.
            - **NominalGEE**: Generalized Estimating Equations for nominal responses.
            - **OrdinalGEE**: Generalized Estimating Equations for ordinal responses.

        **Regression Inference Models**
            - **OLS**: Ordinary Least Squares for linear regression.
            - **WLS**: Weighted Least Squares for cases with non-constant variance.
            - **GLS**: Generalized Least Squares for regression with correlated errors.
            - **RLM**: Robust Linear Models for regression with outliers.
            - **QuantReg**: Quantile Regression for modeling different quantiles.
            - **GLSAR**: GLS with autoregressive error model.
            - **MixedLM**: Mixed Linear Model for hierarchical or longitudinal data.
            - **PHReg**: Proportional Hazards model for survival analysis.

    Scoring Metrics
    +++++++++++++++

        **Classification Scoring Metrics**
            - **Accuracy, Balanced Accuracy**: Measures overall and balanced accuracy.
            - **Average Precision, F1 Score Variants**: Assesses precision-recall balance.
            - **Negative Log Loss**: Negative log-likelihood of the classifier.
            - **Precision, Recall, Jaccard Index**: Evaluates the positive identified samples.
            - **ROC AUC**: Area Under the ROC Curve for model discrimination capability.

        **Regression Scoring Metrics**
            - **Explained Variance, MAE, MSE, RMSE, MSLE, MedAE**: Measures of error and variance explained by the model.
            - **R2, MPD, MGD, MAPE**: Metrics for accuracy and prediction deviation.

    Model Tuners
    ++++++++++++
        - **GridSearchCV**: Exhaustive search over specified parameter values.
        - **RandomizedSearchCV**: Randomized search on hyper parameters.
        - **BayesSearchCV**: Bayesian approach to hyperparameter optimization.

    Default Parameter Grids
    +++++++++++++++++++++++

    We provide default parameter grids for users who do not wish to provide it. Feel free to use the ones below as a template for your own to use with ``predict_ml(custom_param_grids=..)``

    **Parameter Grid for Classification Model Tuning**

    >>> default_param_grids_classification = {
    ...     'LogisticRegression': {
    ...         'C': [0.1, 1, 10, 100],
    ...         'penalty': ['l2'],
    ...         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    ...     },
    ...     'DecisionTreeClassifier': {
    ...         'max_depth': [None, 10, 20, 30, 40, 50],
    ...         'min_samples_split': [2, 5, 10],
    ...         'min_samples_leaf': [1, 2, 4]
    ...     },
    ...     'RandomForestClassifier': {
    ...         'n_estimators': [100, 200, 300, 400],
    ...         'max_features': ['auto', 'sqrt'],
    ...         'max_depth': [None, 10, 20, 30, 40],
    ...         'min_samples_split': [2, 5, 10],
    ...         'min_samples_leaf': [1, 2, 4]
    ...     },
    ...     'GradientBoostingClassifier': {
    ...         'n_estimators': [100, 200, 300],
    ...         'learning_rate': [0.01, 0.1, 0.2, 0.5],
    ...         'max_depth': [3, 5, 7, 9]
    ...     },
    ...     'SVC': {
    ...         'C': [0.1, 1, 10, 100, 1000],
    ...         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    ...         'gamma': ['scale', 'auto']
    ...     },
    ...     'KNeighborsClassifier': {
    ...         'n_neighbors': [3, 5, 7, 9],
    ...         'weights': ['uniform', 'distance'],
    ...         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    ...     }
    ... }


    **Parameter Grid for Regression Model Tuning**

    >>> default_param_grids_regression = {
    ...     'LinearRegression': {
    ...         # Linear Regression usually does not need hyperparameter tuning except for regularization
    ...     },
    ...     'Ridge': {
    ...         'alpha': [0.1, 1.0, 10.0, 100.0],
    ...         'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    ...     },
    ...     'Lasso': {
    ...         'alpha': [0.1, 1.0, 10.0, 100.0],
    ...         'selection': ['cyclic', 'random']
    ...     },
    ...     'DecisionTreeRegressor': {
    ...         'max_depth': [None, 10, 20, 30, 40, 50],
    ...         'min_samples_split': [2, 5, 10],
    ...         'min_samples_leaf': [1, 2, 4]
    ...     },
    ...     'RandomForestRegressor': {
    ...         'n_estimators': [100, 200, 300, 400],
    ...         'max_features': ['auto', 'sqrt'],
    ...         'max_depth': [None, 10, 20, 30, 40],
    ...         'min_samples_split': [2, 5, 10],
    ...         'min_samples_leaf': [1, 2, 4]
    ...     },
    ...     'GradientBoostingRegressor': {
    ...         'n_estimators': [100, 200, 300],
    ...         'learning_rate': [0.01, 0.1, 0.2, 0.5],
    ...         'max_depth': [3, 5, 7, 9]
    ...     },
    ...     'SVR': {
    ...         'C': [0.1, 1, 10, 100, 1000],
    ...         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    ...         'gamma': ['scale', 'auto']
    ...     },
    ...     'KNeighborsRegressor': {
    ...         'n_neighbors': [3, 5, 7, 9],
    ...         'weights': ['uniform', 'distance'],
    ...         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    ...     }
    ... }

    """
    if datetime_transformer is None:
        datetime_transformer = FunctionTransformer(datetime_feature_extractor, validate=False)

    if formula and df is not None:
        # Inference pipeline
        return model_recommendation_core_inference(df, formula, priority_models=priority_models, n_top_models=n_top_models,
                                                   model_kwargs=model_kwargs, verbose=verbose)
    elif x_cols is not None and y_col is not None and df is not None:
        # ML pipeline
        x_train_processed, x_test_processed, y_train, y_test, task_type = data_preprocessing_core(
            df=df,
            x_cols=x_cols,
            y_col=y_col,
            data_state=data_state,
            test_size=test_size,
            random_state=random_state,
            numeric_imputer=numeric_imputer,
            numeric_scaler=numeric_scaler,
            categorical_imputer=categorical_imputer,
            categorical_encoder=categorical_encoder,
            text_vectorizer=text_vectorizer,
            datetime_transformer=datetime_transformer,
            verbose=verbose
        )

        recommended_models = model_recommendation_core(
            x_train=x_train_processed,
            y_train=y_train,
            task_type=task_type,
            cv=cv,
            priority_metrics=priority_metrics,
            n_top_models=n_top_models,
            verbose=verbose
        )

        tuned_models = model_tuning_core(
            x_train=x_train_processed,
            y_train=y_train,
            task_type=task_type,
            models=recommended_models,
            priority_metrics=priority_metrics,
            refit_metric=refit_metric,
            priority_tuners=priority_tuners,
            custom_param_grids=custom_param_grids,
            n_jobs=n_jobs,
            cv=cv,
            n_iter_random=n_iter_random,
            n_iter_bayesian=n_iter_bayesian,
            verbose=verbose,
            random_state=random_state
        )

        return tuned_models
    else:
        raise ValueError("Invalid input: Either provide a formula for inference or x_cols and y_col for machine learning.")
