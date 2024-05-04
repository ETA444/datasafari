import pytest
import numpy as np
import pandas as pd
from datasafari.predictor.predict_ml import data_preprocessing_core


@pytest.fixture
def sample_data_dpc():
    """ Provides sample data for tests related to data_preprocessing_core. """
    return pd.DataFrame({
        'Age': np.random.randint(18, 35, size=100),
        'Salary': np.random.normal(50000, 12000, size=100),
        'Department': np.random.choice(['HR', 'Tech', 'Marketing'], size=100),
        'Review': ['Good review']*50 + ['Bad review']*50,
        'Employment Date': pd.date_range(start='2010-01-01', periods=100, freq='M')
    })


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
