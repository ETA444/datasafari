import pytest
import pandas as pd
import numpy as np
from datasafari.evaluator.evaluate_dtype import evaluate_dtype


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame to use in the tests."""
    return pd.DataFrame({
        'Age': np.random.randint(18, 35, 100),
        'Income': np.random.normal(50000, 15000, 100),
        'Department': np.random.choice(['HR', 'Tech', 'Admin'], 100),
        'EntryDate': pd.date_range(start='1/1/2020', periods=100, freq='M')
    })


# TESTING ERROR-HANDLING #

def test_evaluate_dtype_invalid_dataframe_type():
    """Test passing invalid dataframe type should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_dtype("not a dataframe", ['Age'])


def test_evaluate_dtype_invalid_col_names_type(sample_dataframe):
    """Test passing invalid col_names type should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_dtype(sample_dataframe, "Age")


def test_evaluate_dtype_invalid_col_names_entry(sample_dataframe):
    """Test passing non-string in col_names should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_dtype(sample_dataframe, [123])


def test_evaluate_dtype_invalid_max_unique_values_ratio_type(sample_dataframe):
    """Test passing non-float for max_unique_values_ratio should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_dtype(sample_dataframe, ['Age'], max_unique_values_ratio="0.05")


def test_evaluate_dtype_invalid_min_unique_values_type(sample_dataframe):
    """Test passing non-int for min_unique_values should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_dtype(sample_dataframe, ['Age'], min_unique_values="10")


def test_evaluate_dtype_invalid_string_length_threshold_type(sample_dataframe):
    """Test passing non-int for string_length_threshold should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_dtype(sample_dataframe, ['Department'], string_length_threshold="50")


def test_evaluate_dtype_invalid_output_type(sample_dataframe):
    """Test passing non-string for output should raise TypeError."""
    with pytest.raises(TypeError):
        evaluate_dtype(sample_dataframe, ['Age'], output=123)


def test_evaluate_dtype_empty_dataframe():
    """Test handling of an empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        evaluate_dtype(empty_df, ['Age'])


def test_evaluate_dtype_invalid_max_unique_values_ratio(sample_dataframe):
    """Test invalid max_unique_values_ratio values should raise ValueError."""
    with pytest.raises(ValueError):
        evaluate_dtype(sample_dataframe, ['Age'], max_unique_values_ratio=-0.1)
    with pytest.raises(ValueError):
        evaluate_dtype(sample_dataframe, ['Age'], max_unique_values_ratio=1.1)


def test_evaluate_dtype_invalid_min_unique_values(sample_dataframe):
    """Test invalid min_unique_values should raise ValueError."""
    with pytest.raises(ValueError):
        evaluate_dtype(sample_dataframe, ['Age'], min_unique_values=0)


def test_evaluate_dtype_invalid_string_length_threshold(sample_dataframe):
    """Test invalid string_length_threshold should raise ValueError."""
    with pytest.raises(ValueError):
        evaluate_dtype(sample_dataframe, ['Department'], string_length_threshold=0)


def test_evaluate_dtype_empty_col_names_list(sample_dataframe):
    """Test empty col_names list should raise ValueError."""
    with pytest.raises(ValueError):
        evaluate_dtype(sample_dataframe, [])


def test_evaluate_dtype_nonexistent_column(sample_dataframe):
    """Test handling non-existent column should raise ValueError."""
    with pytest.raises(ValueError):
        evaluate_dtype(sample_dataframe, ['NonexistentColumn'])


def test_evaluate_dtype_invalid_output_option(sample_dataframe):
    """Test invalid output option should raise ValueError."""
    with pytest.raises(ValueError):
        evaluate_dtype(sample_dataframe, ['Age'], output='list_x')


# TESTING FUNCTIONALITY #

def test_evaluate_dtype_numerical_identification(sample_dataframe):
    """Test correct identification of numerical data types."""
    result = evaluate_dtype(sample_dataframe, ['Income'], output='dict')
    assert result['Income'] == 'numerical', "Income should be identified as numerical"


def test_evaluate_dtype_categorical_identification_from_numerical(sample_dataframe):
    """Test identification of numerical data that functions as categorical."""
    sample_dataframe['CodedDepartment'] = sample_dataframe['Department'].astype('category').cat.codes
    result = evaluate_dtype(sample_dataframe, ['CodedDepartment'], output='dict', min_unique_values=3)
    assert result['CodedDepartment'] == 'categorical', "CodedDepartment should be identified as categorical"


def test_evaluate_dtype_text_identification(sample_dataframe):
    """Test correct identification of text data types."""
    sample_dataframe['LongText'] = 'This is a long string ' * 10
    result = evaluate_dtype(sample_dataframe, ['LongText'], output='dict', string_length_threshold=10)
    assert result['LongText'] == 'text', "LongText should be identified as text data"


def test_evaluate_dtype_datetime_identification(sample_dataframe):
    """Test correct identification of datetime data types."""
    result = evaluate_dtype(sample_dataframe, ['EntryDate'], output='dict')
    assert result['EntryDate'] == 'datetime', "EntryDate should be identified as datetime"


def test_evaluate_dtype_multiple_columns(sample_dataframe):
    """Test correct identification for multiple columns."""
    result = evaluate_dtype(sample_dataframe, ['Age', 'Income', 'Department', 'EntryDate'], output='dict')
    expected = {
        'Age': 'numerical',
        'Income': 'numerical',
        'Department': 'categorical',
        'EntryDate': 'datetime'
    }
    assert result == expected, "Multiple columns should be identified correctly"


def test_evaluate_dtype_output_list_numerical(sample_dataframe):
    """Test list output for numerical data identification."""
    result = evaluate_dtype(sample_dataframe, ['Age', 'Income'], output='list_n')
    expected = [True, True]
    assert result == expected, "List of numerical data types should be identified correctly"


def test_evaluate_dtype_output_list_categorical(sample_dataframe):
    """Test list output for categorical data identification."""
    result = evaluate_dtype(sample_dataframe, ['Age', 'Department'], output='list_c')
    expected = [False, True]
    assert result == expected, "List of categorical data types should be identified correctly"


def test_evaluate_dtype_output_list_datetime(sample_dataframe):
    """Test list output for datetime data identification."""
    result = evaluate_dtype(sample_dataframe, ['EntryDate'], output='list_d')
    assert result == [True], "List for datetime data type should be identified correctly"


def test_evaluate_dtype_output_list_text(sample_dataframe):
    """Test list output for text data identification."""
    sample_dataframe['Comments'] = ['This is a very detailed comment about something'] * 100
    result = evaluate_dtype(sample_dataframe, ['Comments'], output='list_t', string_length_threshold=10)
    assert result == [True], "List for text data type should be identified correctly"


def test_evaluate_dtype_small_categorical(sample_dataframe):
    """Test handling of small dataset with categorical data."""
    result = evaluate_dtype(sample_dataframe, ['Department'], output='dict')
    assert result['Department'] == 'categorical', "Small categorical column should be identified as categorical"


def test_evaluate_dtype_small_text(sample_dataframe):
    """Test handling of small dataset with text data."""
    small_df = sample_dataframe.sample(n=60)
    small_df['ShortText'] = ['short'] * 60
    result = evaluate_dtype(small_df, ['ShortText'], output='dict', string_length_threshold=4)
    print(result)
    assert result['ShortText'] == 'text', "Small text column should be identified as text"


def test_evaluate_dtype_small_numerical(sample_dataframe):
    """Test handling of small dataset with numerical data."""
    small_df = sample_dataframe.sample(n=60)
    result = evaluate_dtype(small_df, ['Income'], output='dict')
    print(result)
    assert result['Income'] == 'numerical', "Small numerical column should be identified as numerical"
