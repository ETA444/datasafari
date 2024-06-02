import pytest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from datasafari.predictor.predict_ml import (
    data_preprocessing_core, calculate_composite_score, model_recommendation_core,
    model_tuning_core, model_recommendation_core_inference
)


@pytest.fixture
def sample_data_dpc():
    """ Provides sample data for tests related to data_preprocessing_core(). """
    np.random.seed(42)
    return pd.DataFrame({
        'Age': np.random.randint(18, 35, size=100),
        'Salary': np.random.normal(50000, 12000, size=100),
        'Department': np.random.choice(['HR', 'Tech', 'Marketing'], size=100),
        'Review': ['Good review'] * 50 + ['Bad review'] * 50,
        'Employment Date': pd.date_range(start='2010-01-01', periods=100, freq='M')
    })


@pytest.fixture
def sample_data_ccs():
    """ Provides sample data for tests related to calculate_composite_score(). """
    scores = {'Accuracy': 0.95, 'Precision': 0.90, 'Recall': 0.85}
    metric_weights = {'Accuracy': 5, 'Precision': 3, 'Recall': 2}
    return scores, metric_weights


@pytest.fixture()
def sample_data_mrc_mtc():
    """ Provides sample data for tests of model_recommendation_core() and model_tuning_core(). """
    np.random.seed(42)
    x_train = np.random.rand(100, 5)
    y_train_classification = np.random.randint(0, 2, size=100)
    y_train_regression = np.random.rand(100)
    return x_train, y_train_classification, y_train_regression


@pytest.fixture
def sample_data_mrci():
    """ Provides sample data for tests of model_recommendation_core_inference(). """
    np.random.seed(42)
    df = pd.DataFrame({
        'Age': np.random.randint(20, 70, size=100),
        'Experience': np.random.randint(1, 30, size=100),
        'Salary': np.random.normal(50000, 15000, size=100)
    })
    return df


# TESTING ERROR-HANDLING of data_preprocessing_core() #

def test_invalid_df_type(sample_data_dpc):
    """ Test TypeError is raised when df is not a DataFrame. """
    with pytest.raises(TypeError):
        data_preprocessing_core("not_a_dataframe", ["Age"], "Salary", "unprocessed")


def test_invalid_x_cols_type(sample_data_dpc):
    """ Test TypeError is raised when x_cols is not a list of strings. """
    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, "not_a_list", "Salary", "unprocessed")

    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, [1, 2, 3], "Salary", "unprocessed")


def test_invalid_y_col_type(sample_data_dpc):
    """ Test TypeError is raised when y_col is not a string. """
    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, ["Age"], 123, "unprocessed")


def test_invalid_data_state_type(sample_data_dpc):
    """ Test TypeError is raised when data_state is not a string. """
    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Salary", 123)


def test_invalid_test_size_type(sample_data_dpc):
    """ Test TypeError is raised when test_size is not a float. """
    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Salary", "unprocessed", test_size="0.2")


def test_invalid_test_size_value(sample_data_dpc):
    """ Test ValueError is raised when test_size is not between 0 and 1. """
    with pytest.raises(ValueError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Salary", "unprocessed", test_size=1.5)


def test_invalid_random_state_type(sample_data_dpc):
    """ Test TypeError is raised when random_state is not an integer. """
    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Salary", "unprocessed", random_state="42")


def test_invalid_verbose_type(sample_data_dpc):
    """ Test TypeError is raised when verbose is not an integer. """
    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Salary", "unprocessed", verbose="1")


def test_invalid_imputer_type(sample_data_dpc):
    """ Test TypeError is raised when numeric_imputer does not support 'fit_transform' or 'fit' and 'transform'. """
    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Salary", "unprocessed", numeric_imputer="invalid_imputer")


def test_invalid_scaler_type(sample_data_dpc):
    """ Test TypeError is raised when numeric_scaler does not support 'fit_transform'. """
    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Salary", "unprocessed", numeric_scaler="invalid_scaler")


def test_invalid_encoder_type(sample_data_dpc):
    """ Test TypeError is raised when categorical_encoder does not support 'fit_transform'. """
    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Salary", "unprocessed", categorical_encoder="invalid_encoder")


def test_invalid_vectorizer_type(sample_data_dpc):
    """ Test TypeError is raised when text_vectorizer does not support 'fit_transform'. """
    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Salary", "unprocessed", text_vectorizer="invalid_vectorizer")


def test_invalid_datetime_transformer_type(sample_data_dpc):
    """ Test TypeError is raised when datetime_transformer does not support 'fit_transform'. """
    with pytest.raises(TypeError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Salary", "unprocessed", datetime_transformer="invalid_transformer")


def test_empty_dataframe(sample_data_dpc):
    """ Test ValueError is raised when df is empty. """
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        data_preprocessing_core(empty_df, ["Age"], "Salary", "unprocessed")


def test_invalid_data_state_value(sample_data_dpc):
    """ Test ValueError is raised when data_state is not 'unprocessed' or 'preprocessed'. """
    with pytest.raises(ValueError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Salary", "invalid_state")


def test_missing_y_col(sample_data_dpc):
    """ Test ValueError is raised when y_col is not found in df. """
    with pytest.raises(ValueError):
        data_preprocessing_core(sample_data_dpc, ["Age"], "Invalid Column", "unprocessed")


def test_missing_x_cols(sample_data_dpc):
    """ Test ValueError is raised when some x_cols are not found in df. """
    with pytest.raises(ValueError):
        data_preprocessing_core(sample_data_dpc, ["Age", "Invalid Column"], "Salary", "unprocessed")


def test_test_size_too_large(sample_data_dpc):
    """ Test ValueError is raised when df does not contain enough data to split according to test_size. """
    small_df = sample_data_dpc.head(3)
    with pytest.raises(ValueError):
        data_preprocessing_core(small_df, ["Age"], "Salary", "unprocessed", test_size=0.5)


# TESTING FUNCTIONALITY of data_preprocessing_core() #

def test_data_preprocessing_default(sample_data_dpc):
    """ Test default functionality of data_preprocessing_core with mixed data types. """
    x_train, x_test, y_train, y_test, task_type = data_preprocessing_core(sample_data_dpc, ["Age", "Salary", "Department", "Review", "Employment Date"], "Salary", "unprocessed")
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in number of columns between training and test sets"
    assert len(x_train) == 80 and len(x_test) == 20, "Incorrect train-test split"
    assert task_type == "regression", "Incorrect task type detected"


def test_data_preprocessing_classification(sample_data_dpc):
    """ Test functionality of data_preprocessing_core for classification task. """
    x_train, x_test, y_train, y_test, task_type = data_preprocessing_core(sample_data_dpc, ["Age", "Salary", "Department", "Review", "Employment Date"], "Department", "unprocessed")
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in number of columns between training and test sets"
    assert task_type == "classification", "Incorrect task type detected"


def test_data_preprocessing_preprocessed(sample_data_dpc):
    """ Test functionality of data_preprocessing_core with preprocessed data. """
    x_train, x_test, y_train, y_test, task_type = data_preprocessing_core(sample_data_dpc, ["Age", "Salary", "Department", "Review", "Employment Date"], "Salary", "preprocessed")
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in number of columns between training and test sets"
    assert len(x_train) == 80 and len(x_test) == 20, "Incorrect train-test split"
    assert task_type == "regression", "Incorrect task type detected"


def test_data_preprocessing_custom_numeric_imputer(sample_data_dpc):
    """ Test functionality of data_preprocessing_core with custom numeric imputer. """
    custom_imputer = SimpleImputer(strategy='mean')
    x_train, x_test, y_train, y_test, task_type = data_preprocessing_core(
        sample_data_dpc, ["Age", "Salary", "Department", "Review", "Employment Date"],
        "Salary", "unprocessed", numeric_imputer=custom_imputer)
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in number of columns between training and test sets"
    assert task_type == "regression", "Incorrect task type detected"


def test_data_preprocessing_custom_categorical_encoder(sample_data_dpc):
    """ Test functionality of data_preprocessing_core with custom categorical encoder. """
    custom_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
    x_train, x_test, y_train, y_test, task_type = data_preprocessing_core(
        sample_data_dpc, ["Age", "Salary", "Department", "Review", "Employment Date"],
        "Salary", "unprocessed", categorical_encoder=custom_encoder)
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in number of columns between training and test sets"
    assert task_type == "regression", "Incorrect task type detected"


def test_data_preprocessing_datetime_features(sample_data_dpc):
    """ Test functionality of data_preprocessing_core with datetime features. """
    x_train, x_test, y_train, y_test, task_type = data_preprocessing_core(
        sample_data_dpc, ["Employment Date"], "Salary", "unprocessed")
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in number of columns between training and test sets"
    assert x_train.shape[1] == 3, "Incorrect number of columns for datetime features"
    assert task_type == "regression", "Incorrect task type detected"


def test_data_preprocessing_text_features(sample_data_dpc):
    """ Test functionality of data_preprocessing_core with text features. """
    x_train, x_test, y_train, y_test, task_type = data_preprocessing_core(
        sample_data_dpc, ["Review"], "Salary", "unprocessed")
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in number of columns between training and test sets"
    assert task_type == "regression", "Incorrect task type detected"


# TESTING ERROR-HANDLING of calculate_composite_score() #

def test_calculate_composite_score_invalid_types():
    """ Test calculate_composite_score with invalid types. """
    with pytest.raises(TypeError, match="Both 'scores' and 'metric_weights' must be dictionaries."):
        calculate_composite_score("invalid", {'Accuracy': 5})


def test_calculate_composite_score_empty_dict():
    """ Test calculate_composite_score with empty dictionary inputs. """
    with pytest.raises(ValueError, match="'scores' and 'metric_weights' dictionaries cannot be empty."):
        calculate_composite_score({}, {'Accuracy': 5})


def test_calculate_composite_score_missing_weights(sample_data_ccs):
    """ Test calculate_composite_score with missing weights for some metrics. """
    scores, metric_weights = sample_data_ccs
    del metric_weights['Recall']
    with pytest.raises(ValueError, match="Missing metric weights for: Recall"):
        calculate_composite_score(scores, metric_weights)


def test_calculate_composite_score_zero_weights():
    """ Test calculate_composite_score with all zero weights. """
    with pytest.raises(ValueError):
        calculate_composite_score({'Accuracy': 0.95}, {'Accuracy': 0})


# TESTING FUNCTIONALITY for calculate_composite_score() #

def test_calculate_composite_score_basic(sample_data_ccs):
    """ Test calculate_composite_score with basic valid input. """
    scores, metric_weights = sample_data_ccs
    result = calculate_composite_score(scores, metric_weights)
    assert pytest.approx(result, 0.01) == 0.92


def test_calculate_composite_score_zero_weights_extra():
    """ Test calculate_composite_score with some zero weights. """
    scores = {'Accuracy': 0.95, 'Precision': 0.90, 'Recall': 0.85}
    metric_weights = {'Accuracy': 5, 'Precision': 0, 'Recall': 2}
    result = calculate_composite_score(scores, metric_weights)
    assert pytest.approx(result, 0.01) == 0.92


# TESTING ERROR-HANDLING for model_recommendation_core() #

def test_model_recommendation_core_type_error_x_train(sample_data_mrc_mtc):
    """ Test model_recommendation_core TypeError when x_train is not DataFrame or ndarray. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(TypeError, match="model_recommendation_core\\(\\): 'x_train' must be a pandas DataFrame or NumPy ndarray."):
        model_recommendation_core("invalid_input", y_train, task_type="classification")


def test_model_recommendation_core_type_error_y_train(sample_data_mrc_mtc):
    """ Test model_recommendation_core TypeError when y_train is not Series or ndarray. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(TypeError, match="model_recommendation_core\\(\\): 'y_train' must be a pandas Series or NumPy ndarray."):
        model_recommendation_core(x_train, "invalid_input", task_type="classification")


def test_model_recommendation_core_type_error_task_type(sample_data_mrc_mtc):
    """ Test model_recommendation_core ValueError when task_type is not 'classification' or 'regression'. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): 'task_type' must be either 'classification' or 'regression'."):
        model_recommendation_core(x_train, y_train, task_type="invalid")


def test_model_recommendation_core_type_error_priority_metrics(sample_data_mrc_mtc):
    """ Test model_recommendation_core TypeError when priority_metrics is not a list. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(TypeError, match="model_recommendation_core\\(\\): 'priority_metrics' must be a list of scoring metric names."):
        model_recommendation_core(x_train, y_train, task_type="classification", priority_metrics="invalid")


def test_model_recommendation_core_type_error_cv(sample_data_mrc_mtc):
    """ Test model_recommendation_core TypeError when cv is not an integer. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(TypeError, match="model_recommendation_core\\(\\): 'cv' must be an integer."):
        model_recommendation_core(x_train, y_train, task_type="classification", cv="invalid")


def test_model_recommendation_core_type_error_n_top_models(sample_data_mrc_mtc):
    """ Test model_recommendation_core ValueError when n_top_models is not an integer greater than 0. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): 'n_top_models' must be an integer greater than 0."):
        model_recommendation_core(x_train, y_train, task_type="classification", n_top_models=0)


def test_model_recommendation_core_type_error_verbose(sample_data_mrc_mtc):
    """ Test model_recommendation_core TypeError when verbose is not an integer. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(TypeError, match="model_recommendation_core\\(\\): 'verbose' must be an integer value."):
        model_recommendation_core(x_train, y_train, task_type="classification", verbose="invalid")


def test_model_recommendation_core_value_error_shape(sample_data_mrc_mtc):
    """ Test model_recommendation_core ValueError when x_train and y_train have incompatible shapes. """
    x_train, _, y_train = sample_data_mrc_mtc
    y_train = y_train[:-1]  # mismatched shapes
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): 'x_train' and 'y_train' must have the same number of rows."):
        model_recommendation_core(x_train, y_train, task_type="classification")


def test_model_recommendation_core_value_error_empty_x_train(sample_data_mrc_mtc):
    """Test model_recommendation_core ValueError when x_train is empty."""
    _, _, y_train = sample_data_mrc_mtc
    x_train = np.empty((0, 5))  # Properly emptying x_train
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): 'x_train' cannot be empty."):
        model_recommendation_core(x_train, y_train, task_type="classification")


def test_model_recommendation_core_value_error_empty_y_train(sample_data_mrc_mtc):
    """Test model_recommendation_core ValueError when y_train is empty."""
    x_train, _, _ = sample_data_mrc_mtc
    y_train = np.empty((0,))  # Properly emptying y_train
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): 'y_train' cannot be empty."):
        model_recommendation_core(x_train, y_train, task_type="classification")


def test_model_recommendation_core_value_error_priority_metrics_duplicates(sample_data_mrc_mtc):
    """ Test model_recommendation_core ValueError when priority_metrics contains duplicate values. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): 'priority_metrics' should not contain duplicate values."):
        model_recommendation_core(x_train, y_train, task_type="classification", priority_metrics=["Accuracy", "Accuracy"])


def test_model_recommendation_core_value_error_priority_metrics_not_strings(sample_data_mrc_mtc):
    """ Test model_recommendation_core ValueError when priority_metrics contains non-string items. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): All items in 'priority_metrics' must be strings representing metric names."):
        model_recommendation_core(x_train, y_train, task_type="classification", priority_metrics=["Accuracy", 123])


def test_model_recommendation_core_value_error_invalid_priority_metrics(sample_data_mrc_mtc):
    """ Test model_recommendation_core ValueError when priority_metrics contains invalid metric names. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): Invalid metric\\(s\\) in 'priority_metrics': .+"):
        model_recommendation_core(x_train, y_train, task_type="classification", priority_metrics=["InvalidMetric"])


def test_model_recommendation_core_value_error_invalid_metrics_classification(sample_data_mrc_mtc):
    """ Test model_recommendation_core ValueError when priority_metrics contains invalid metrics for classification. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): The following priority metrics are not valid for classification: .+"):
        model_recommendation_core(x_train, y_train, task_type="classification", priority_metrics=["neg_mean_squared_error"])


def test_model_recommendation_core_value_error_invalid_metrics_regression(sample_data_mrc_mtc):
    """ Test model_recommendation_core ValueError when priority_metrics contains invalid metrics for regression. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): The following priority metrics are not valid for regression:"):
        model_recommendation_core(x_train, y_train, task_type="regression", priority_metrics=["accuracy"])


def test_model_recommendation_core_value_error_n_top_models_exceeds_classification(sample_data_mrc_mtc):
    """ Test model_recommendation_core ValueError when n_top_models exceeds available models for classification. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): 'n_top_models' cannot exceed the number of available classification models .+"):
        model_recommendation_core(x_train, y_train, task_type="classification", n_top_models=10)


def test_model_recommendation_core_value_error_n_top_models_exceeds_regression(sample_data_mrc_mtc):
    """ Test model_recommendation_core ValueError when n_top_models exceeds available models for regression. """
    x_train, _, y_train = sample_data_mrc_mtc
    with pytest.raises(ValueError, match="model_recommendation_core\\(\\): 'n_top_models' cannot exceed the number of available regression models .+"):
        model_recommendation_core(x_train, y_train, task_type="regression", n_top_models=10)


# TESTING FUNCTIONALITY of model_recommendation_core() #

def test_model_recommendation_core_classification(sample_data_mrc_mtc):
    """ Test model_recommendation_core functionality for classification task. """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    recommended_models = model_recommendation_core(x_train, y_train_classification, task_type="classification")
    assert isinstance(recommended_models, dict)
    assert len(recommended_models) <= 3
    for model_name, model_obj in recommended_models.items():
        assert isinstance(model_name, str)
        assert hasattr(model_obj, "fit")
        assert hasattr(model_obj, "predict")


def test_model_recommendation_core_regression(sample_data_mrc_mtc):
    """ Test model_recommendation_core functionality for regression task. """
    x_train, _, y_train_regression = sample_data_mrc_mtc
    recommended_models = model_recommendation_core(x_train, y_train_regression, task_type="regression", priority_metrics=["neg_mean_squared_error"])
    assert isinstance(recommended_models, dict)
    assert len(recommended_models) <= 3
    for model_name, model_obj in recommended_models.items():
        assert isinstance(model_name, str)
        assert hasattr(model_obj, "fit")
        assert hasattr(model_obj, "predict")


def test_model_recommendation_core_classification_no_priority(sample_data_mrc_mtc):
    """ Test model_recommendation_core functionality for classification task without priority metrics. """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    recommended_models = model_recommendation_core(x_train, y_train_classification, task_type="classification")
    assert isinstance(recommended_models, dict)
    assert len(recommended_models) <= 3
    for model_name, model_obj in recommended_models.items():
        assert isinstance(model_name, str)
        assert hasattr(model_obj, "fit")
        assert hasattr(model_obj, "predict")


def test_model_recommendation_core_regression_no_priority(sample_data_mrc_mtc):
    """ Test model_recommendation_core functionality for regression task without priority metrics. """
    x_train, _, y_train_regression = sample_data_mrc_mtc
    recommended_models = model_recommendation_core(x_train, y_train_regression, task_type="regression")
    assert isinstance(recommended_models, dict)
    assert len(recommended_models) <= 3
    for model_name, model_obj in recommended_models.items():
        assert isinstance(model_name, str)
        assert hasattr(model_obj, "fit")
        assert hasattr(model_obj, "predict")


def test_model_recommendation_core_classification_verbose(sample_data_mrc_mtc, capsys):
    """ Test model_recommendation_core functionality for classification task with verbose output. """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    model_recommendation_core(x_train, y_train_classification, task_type="classification", priority_metrics=["accuracy"], verbose=2)
    captured = capsys.readouterr()
    assert "< MODEL RECOMMENDATIONS >" in captured.out


def test_model_recommendation_core_regression_verbose(sample_data_mrc_mtc, capsys):
    """ Test model_recommendation_core functionality for regression task with verbose output. """
    x_train, _, y_train_regression = sample_data_mrc_mtc
    model_recommendation_core(x_train, y_train_regression, task_type="regression", priority_metrics=["neg_mean_squared_error"], verbose=2)
    captured = capsys.readouterr()
    assert "< MODEL RECOMMENDATIONS >" in captured.out


# TESTING ERROR-HANDLING of model_tuning_core() #

def test_model_tuning_core_invalid_x_train_type(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises TypeError when 'x_train' is not a DataFrame or ndarray.
    """
    _, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(TypeError):
        model_tuning_core(x_train="invalid_type", y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()})


def test_model_tuning_core_invalid_y_train_type(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises TypeError when 'y_train' is not a Series or ndarray.
    """
    x_train, _, _ = sample_data_mrc_mtc
    with pytest.raises(TypeError):
        model_tuning_core(x_train=x_train, y_train="invalid_type", task_type='classification', models={'model': LogisticRegression()})


def test_model_tuning_core_invalid_task_type():
    """
    Tests if model_tuning_core raises TypeError when 'task_type' is not a string.
    """
    x_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    with pytest.raises(TypeError):
        model_tuning_core(x_train=x_train, y_train=y_train, task_type=123, models={'model': LogisticRegression()})

    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train, task_type='invalid', models={'model': LogisticRegression()})


def test_model_tuning_core_invalid_models_type(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises TypeError when 'models' is not a dictionary.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(TypeError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models="invalid_type")


def test_model_tuning_core_invalid_priority_metrics_type(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises TypeError when 'priority_metrics' is not a list.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(TypeError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, priority_metrics="invalid_type")


def test_model_tuning_core_invalid_priority_tuners_type(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises TypeError when 'priority_tuners' is not a list.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(TypeError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, priority_tuners="invalid_type")


def test_model_tuning_core_invalid_custom_param_grids_type(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises TypeError when 'custom_param_grids' is not a dictionary.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(TypeError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, custom_param_grids="invalid_type")


def test_model_tuning_core_invalid_n_jobs_type(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises TypeError when 'n_jobs' is not an integer.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(TypeError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, n_jobs="invalid_type")


def test_model_tuning_core_invalid_cv_type(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises TypeError when 'cv' is not an integer.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(TypeError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, cv="invalid_type")


def test_model_tuning_core_invalid_cv_value(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises ValueError when 'cv' is less than 1.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, cv=0)


def test_model_tuning_core_invalid_n_iter_random_value(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises ValueError when 'n_iter_random' is less than 1.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, n_iter_random=-1)


def test_model_tuning_core_invalid_n_iter_bayesian_value(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises ValueError when 'n_iter_bayesian' is less than 1.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, n_iter_bayesian=-1)


def test_model_tuning_core_invalid_verbose_type(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises TypeError when 'verbose' is not an integer.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(TypeError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, verbose="invalid_type")


def test_model_tuning_core_invalid_random_state_type(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises TypeError when 'random_state' is not an integer.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(TypeError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, random_state="invalid_type")


def test_model_tuning_core_mismatched_x_y_shapes(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises ValueError when 'x_train' and 'y_train' have different number of rows.
    """
    x_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 99)
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train, task_type='classification', models={'model': LogisticRegression()})


def test_model_tuning_core_empty_x_train(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises ValueError when 'x_train' is empty.
    """
    x_train = np.array([]).reshape(0, 5)
    y_train = np.random.randint(0, 2, 100)
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train, task_type='classification', models={'model': LogisticRegression()})


def test_model_tuning_core_empty_y_train(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises ValueError when 'y_train' is empty.
    """
    x_train = np.random.randn(100, 5)
    y_train = np.array([])
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train, task_type='classification', models={'model': LogisticRegression()})


def test_model_tuning_core_duplicate_priority_metrics(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises ValueError when 'priority_metrics' contains duplicate values.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, priority_metrics=["accuracy", "accuracy"])


def test_model_tuning_core_invalid_priority_metrics_for_task_type(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises ValueError when 'priority_metrics' contains invalid metrics for the specified 'task_type'.
    """
    x_train, _, y_train_regression = sample_data_mrc_mtc
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train_regression, task_type='regression', models={'model': LinearRegression()}, priority_metrics=["accuracy"])

    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, priority_metrics=["r2"])


def test_model_tuning_core_invalid_priority_tuners(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises ValueError when 'priority_tuners' contains unrecognized tuner names.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, priority_tuners=["invalid"])


def test_model_tuning_core_invalid_refit_metric(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core raises ValueError when 'refit_metric' is not applicable to the provided 'task_type'.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train_classification, task_type='classification', models={'model': LogisticRegression()}, refit_metric="r2")

    _, y_train_regression, _ = sample_data_mrc_mtc
    with pytest.raises(ValueError):
        model_tuning_core(x_train=x_train, y_train=y_train_regression, task_type='regression', models={'model': LinearRegression()}, refit_metric="accuracy")


# TESTING FUNCTIONALITY of model_tuning_core() #

def test_model_tuning_core_classification_with_logistic_regression(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core works for classification with Logistic Regression model.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    models = {'LogisticRegression': LogisticRegression()}
    tuned_models = model_tuning_core(x_train, y_train_classification, 'classification', models)
    assert 'LogisticRegression' in tuned_models
    assert isinstance(tuned_models['LogisticRegression']['best_model'], LogisticRegression)


def test_model_tuning_core_regression_with_ridge(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core works for regression with Linear Regression model.
    """
    x_train, _, y_train_regression = sample_data_mrc_mtc
    models = {'Ridge': Ridge()}
    tuned_models = model_tuning_core(x_train, y_train_regression, 'regression', models)
    assert 'Ridge' in tuned_models
    assert isinstance(tuned_models['Ridge']['best_model'], Ridge)


def test_model_tuning_core_classification_with_svc(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core works for classification with SVC model.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    models = {'SVC': SVC(probability=True)}
    tuned_models = model_tuning_core(x_train, y_train_classification, 'classification', models, priority_tuners=['grid'])
    assert 'SVC' in tuned_models
    assert isinstance(tuned_models['SVC']['best_model'], SVC)


def test_model_tuning_core_regression_with_svr(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core works for regression with SVR model.
    """
    x_train, _, y_train_regression = sample_data_mrc_mtc
    models = {'SVR': SVR()}
    tuned_models = model_tuning_core(x_train, y_train_regression, 'regression', models, priority_tuners=['grid'])
    assert 'SVR' in tuned_models
    assert isinstance(tuned_models['SVR']['best_model'], SVR)


def test_model_tuning_core_classification_with_priority_metrics(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core works for classification with priority metrics.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    models = {'LogisticRegression': LogisticRegression()}
    tuned_models = model_tuning_core(x_train, y_train_classification, 'classification', models, priority_metrics=['accuracy', 'f1_micro'])
    assert 'LogisticRegression' in tuned_models
    assert isinstance(tuned_models['LogisticRegression']['best_model'], LogisticRegression)


def test_model_tuning_core_regression_with_priority_metrics(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core works for regression with priority metrics.
    """
    x_train, _, y_train_regression = sample_data_mrc_mtc
    models = {'Ridge': Ridge()}
    tuned_models = model_tuning_core(x_train, y_train_regression, 'regression', models, priority_metrics=['r2', 'neg_mean_squared_error'])
    assert 'Ridge' in tuned_models
    assert isinstance(tuned_models['Ridge']['best_model'], Ridge)


def test_model_tuning_core_classification_with_custom_param_grids(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core works for classification with custom parameter grids.
    """
    x_train, y_train_classification, _ = sample_data_mrc_mtc
    models = {'LogisticRegression': LogisticRegression()}
    custom_param_grids = {
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'sag']
        }
    }
    tuned_models = model_tuning_core(x_train, y_train_classification, 'classification', models, custom_param_grids=custom_param_grids)
    assert 'LogisticRegression' in tuned_models
    assert isinstance(tuned_models['LogisticRegression']['best_model'], LogisticRegression)


def test_model_tuning_core_regression_with_custom_param_grids(sample_data_mrc_mtc):
    """
    Tests if model_tuning_core works for regression with custom parameter grids.
    """
    x_train, _, y_train_regression = sample_data_mrc_mtc
    models = {'Ridge': Ridge()}
    custom_param_grids = {
        'Ridge': {
            'solver': ['svd', 'cholesky']
        }
    }
    tuned_models = model_tuning_core(x_train, y_train_regression, 'regression', models, custom_param_grids=custom_param_grids)
    assert 'Ridge' in tuned_models
    assert isinstance(tuned_models['Ridge']['best_model'], Ridge)


# TESTING ERROR-HANDLING of model_recommendation_core_inference() #

def test_model_recommendation_core_inference_invalid_df_type():
    """
    Tests if model_recommendation_core_inference raises a TypeError for invalid df type.
    """
    with pytest.raises(TypeError, match=r"model_recommendation_core_inference\(\): 'df' must be a pandas DataFrame."):
        model_recommendation_core_inference(df=[], formula='Salary ~ Age + Experience')


def test_model_recommendation_core_inference_invalid_formula_type(sample_data_mrci):
    """
    Tests if model_recommendation_core_inference raises a TypeError for invalid formula type.
    """
    df = sample_data_mrci
    with pytest.raises(TypeError, match=r"model_recommendation_core_inference\(\): 'formula' must be a string."):
        model_recommendation_core_inference(df, formula=123)


def test_model_recommendation_core_inference_invalid_priority_models_type(sample_data_mrci):
    """
    Tests if model_recommendation_core_inference raises a TypeError for invalid priority_models type.
    """
    df = sample_data_mrci
    with pytest.raises(TypeError, match=r"model_recommendation_core_inference\(\): 'priority_models' must be a list of strings."):
        model_recommendation_core_inference(df, 'Salary ~ Age + Experience', priority_models='Logit')


def test_model_recommendation_core_inference_invalid_n_top_models_type(sample_data_mrci):
    """
    Tests if model_recommendation_core_inference raises a ValueError for invalid n_top_models type.
    """
    df = sample_data_mrci
    with pytest.raises(ValueError, match=r"model_recommendation_core_inference\(\): 'n_top_models' must be an integer greater than 0."):
        model_recommendation_core_inference(df, 'Salary ~ Age + Experience', n_top_models=-1)


def test_model_recommendation_core_inference_invalid_model_kwargs_type(sample_data_mrci):
    """
    Tests if model_recommendation_core_inference raises a TypeError for invalid model_kwargs type.
    """
    df = sample_data_mrci
    with pytest.raises(TypeError, match=r"model_recommendation_core_inference\(\): 'model_kwargs' must be a dictionary."):
        model_recommendation_core_inference(df, 'Salary ~ Age + Experience', model_kwargs='kwargs')


def test_model_recommendation_core_inference_invalid_verbose_type(sample_data_mrci):
    """
    Tests if model_recommendation_core_inference raises a TypeError for invalid verbose type.
    """
    df = sample_data_mrci
    with pytest.raises(TypeError, match=r"model_recommendation_core_inference\(\): 'verbose' must be an integer."):
        model_recommendation_core_inference(df, 'Salary ~ Age + Experience', verbose='loud')


def test_model_recommendation_core_inference_empty_dataframe():
    """
    Tests if model_recommendation_core_inference raises a ValueError for empty dataframe.
    """
    df = pd.DataFrame()
    with pytest.raises(ValueError, match=r"model_recommendation_core_inference\(\): The input DataFrame is empty."):
        model_recommendation_core_inference(df, 'Salary ~ Age + Experience')


def test_model_recommendation_core_inference_invalid_formula_format(sample_data_mrci):
    """
    Tests if model_recommendation_core_inference raises a ValueError for incorrect formula format.
    """
    df = sample_data_mrci
    with pytest.raises(ValueError, match=r"model_recommendation_core_inference\(\): 'formula' must include exactly one '~' to separate dependent and independent variables."):
        model_recommendation_core_inference(df, 'Salary ~ Age ~ Experience')


def test_model_recommendation_core_inference_missing_target_variable(sample_data_mrci):
    """
    Tests if model_recommendation_core_inference raises a ValueError when the target variable is missing from the dataframe.
    """
    df = sample_data_mrci.drop(columns='Salary')
    with pytest.raises(ValueError, match=r"model_recommendation_core_inference\(\): Specified target variable 'Salary' is not in DataFrame."):
        model_recommendation_core_inference(df, 'Salary ~ Age + Experience')


def test_model_recommendation_core_inference_missing_independent_variable(sample_data_mrci):
    """
    Tests if model_recommendation_core_inference raises a ValueError when an independent variable is missing from the dataframe.
    """
    df = sample_data_mrci.drop(columns='Age')
    with pytest.raises(ValueError, match=r"model_recommendation_core_inference\(\): The following independent variables are not in DataFrame: Age."):
        model_recommendation_core_inference(df, 'Salary ~ Age + Experience')


# TESTING FUNCTIONALITY of model_recommendation_core_inference() #

def test_model_recommendation_core_inference_basic_regression(sample_data_mrci):
    """
    Tests the basic functionality of model_recommendation_core_inference for a regression task.
    """
    df = sample_data_mrci
    formula = 'Salary ~ Age + Experience'
    results = model_recommendation_core_inference(df, formula, n_top_models=2, verbose=1)

    assert len(results) <= 2, "Returned more models than requested."
    for model_name, details in results.items():
        assert 'model' in details, f"Model object missing for {model_name}."
        assert 'metrics' in details, f"Metrics missing for {model_name}."

        available_metrics = details['metrics'].keys()
        expected_metrics = ['AIC', 'R-squared']
        for metric in expected_metrics:
            if metric in available_metrics:
                assert metric in details['metrics'], f"{metric} metric missing for {model_name}"


def test_model_recommendation_core_inference_basic_classification():
    """
    Tests the basic functionality of model_recommendation_core_inference for a classification task.
    """
    df = pd.DataFrame({
        'Outcome': np.random.choice([0, 1], size=100),
        'Feature1': np.random.normal(size=100),
        'Feature2': np.random.normal(size=100)
    })
    formula = 'Outcome ~ Feature1 + Feature2'
    results = model_recommendation_core_inference(df, formula, n_top_models=2, verbose=1)

    assert len(results) <= 2, "Returned more models than requested."
    for model_name, details in results.items():
        assert 'model' in details, f"Model object missing for {model_name}."
        assert 'metrics' in details, f"Metrics missing for {model_name}."
        assert 'AIC' in details['metrics'], f"AIC metric missing for {model_name}."
        assert 'Pseudo R-squared' in details['metrics'], f"Pseudo R-squared metric missing for {model_name}."


def test_model_recommendation_core_inference_custom_priority_models(sample_data_mrci):
    """
    Tests the functionality of model_recommendation_core_inference with custom priority models.
    """
    df = sample_data_mrci
    formula = 'Salary ~ Age + Experience'
    priority_models = ['OLS', 'RLM']
    results = model_recommendation_core_inference(df, formula, priority_models=priority_models, verbose=1)

    assert set(results.keys()) <= set(priority_models), "Returned models not in the priority list."


def test_model_recommendation_core_inference_verbose_levels(sample_data_mrci):
    """
    Tests the functionality of model_recommendation_core_inference with different verbosity levels.
    """
    df = sample_data_mrci
    formula = 'Salary ~ Age + Experience'
    results = model_recommendation_core_inference(df, formula, verbose=0)

    assert len(results) > 0, "Function should return at least one model even with verbose=0."

    results = model_recommendation_core_inference(df, formula, verbose=2)
    assert len(results) > 0, "Function should return at least one model even with verbose=2."
    for details in results.values():
        assert 'model' in details, "Model object missing."
        assert 'metrics' in details, "Metrics missing."
