import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for tests
import matplotlib.pyplot as plt

from autofillgluon.utils.utils import (
    calculate_missingness_statistics,
    generate_missingness_pattern,
    evaluate_imputation_accuracy,
    plot_imputation_evaluation
)

# --- Fixtures ---

@pytest.fixture
def sample_df_mixed():
    """DataFrame with mixed types and missing values."""
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [1.1, 2.2, 3.3, np.nan, 5.5],
        'C': ['x', 'y', 'z', 'x', np.nan],
        'D': [True, False, True, False, True]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_df_no_missing():
    """DataFrame with no missing values."""
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_df_all_missing():
    """DataFrame with all missing values in one column."""
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [np.nan, np.nan, np.nan, np.nan, np.nan]
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_df():
    """Empty DataFrame."""
    return pd.DataFrame()

# --- Tests for calculate_missingness_statistics ---

def test_calculate_missingness_stats_mixed(sample_df_mixed):
    """Test with a DataFrame containing mixed types and NaNs."""
    stats = calculate_missingness_statistics(sample_df_mixed)
    expected_stats = {
        'A': {'percent_missing': 20.0, 'count_missing': 1, 'total_rows': 5},
        'B': {'percent_missing': 20.0, 'count_missing': 1, 'total_rows': 5},
        'C': {'percent_missing': 20.0, 'count_missing': 1, 'total_rows': 5},
        'D': {'percent_missing': 0.0, 'count_missing': 0, 'total_rows': 5}
    }
    assert stats == expected_stats

def test_calculate_missingness_stats_no_missing(sample_df_no_missing):
    """Test with a DataFrame containing no NaNs."""
    stats = calculate_missingness_statistics(sample_df_no_missing)
    expected_stats = {
        'A': {'percent_missing': 0.0, 'count_missing': 0, 'total_rows': 5},
        'B': {'percent_missing': 0.0, 'count_missing': 0, 'total_rows': 5}
    }
    assert stats == expected_stats

def test_calculate_missingness_stats_all_missing(sample_df_all_missing):
    """Test with a DataFrame containing a column with all NaNs."""
    stats = calculate_missingness_statistics(sample_df_all_missing)
    expected_stats = {
        'A': {'percent_missing': 0.0, 'count_missing': 0, 'total_rows': 5},
        'B': {'percent_missing': 100.0, 'count_missing': 5, 'total_rows': 5}
    }
    assert stats == expected_stats

def test_calculate_missingness_stats_empty(empty_df):
    """Test with an empty DataFrame."""
    stats = calculate_missingness_statistics(empty_df)
    expected_stats = {}
    assert stats == expected_stats


# --- Tests for generate_missingness_pattern ---

def test_generate_missingness_pattern_percentage(sample_df_no_missing):
    """Test if the correct percentage of missing values is generated."""
    percent_missing = 0.3
    df_missing = generate_missingness_pattern(sample_df_no_missing, percent_missing=percent_missing, random_state=42)
    
    total_elements = sample_df_no_missing.size
    expected_missing_count = int(total_elements * percent_missing)
    actual_missing_count = df_missing.isnull().sum().sum()
    
    # Allow for slight variation due to random sampling
    assert abs(actual_missing_count - expected_missing_count) <= 1 
    assert not df_missing.equals(sample_df_no_missing) # Ensure some values were changed

def test_generate_missingness_pattern_reproducibility(sample_df_no_missing):
    """Test if the pattern is reproducible with the same random_state."""
    df_missing1 = generate_missingness_pattern(sample_df_no_missing, percent_missing=0.2, random_state=123)
    df_missing2 = generate_missingness_pattern(sample_df_no_missing, percent_missing=0.2, random_state=123)
    assert df_missing1.equals(df_missing2)

def test_generate_missingness_pattern_different_state(sample_df_no_missing):
    """Test if different random_states produce different patterns."""
    df_missing1 = generate_missingness_pattern(sample_df_no_missing, percent_missing=0.2, random_state=1)
    df_missing2 = generate_missingness_pattern(sample_df_no_missing, percent_missing=0.2, random_state=2)
    assert not df_missing1.equals(df_missing2)

def test_generate_missingness_pattern_zero_percent(sample_df_no_missing):
    """Test generating missingness with 0%."""
    df_missing = generate_missingness_pattern(sample_df_no_missing, percent_missing=0.0, random_state=42)
    # Compare values using numpy's testing functions which handle types better
    np.testing.assert_array_equal(df_missing.values, sample_df_no_missing.values)
    # Also check dtypes explicitly if necessary, but value comparison is often sufficient
    # assert df_missing.dtypes.equals(sample_df_no_missing.dtypes) 
    assert df_missing.isnull().sum().sum() == 0

def test_generate_missingness_pattern_hundred_percent(sample_df_no_missing):
    """Test generating missingness with 100%."""
    df_missing = generate_missingness_pattern(sample_df_no_missing, percent_missing=1.0, random_state=42)
    assert df_missing.isnull().all().all() # Check if all values are NaN

def test_generate_missingness_pattern_empty(empty_df):
    """Test with an empty DataFrame."""
    df_missing = generate_missingness_pattern(empty_df, percent_missing=0.2, random_state=42)
    assert df_missing.empty
    assert df_missing.equals(empty_df)


# --- Fixtures for evaluate_imputation_accuracy ---

@pytest.fixture
def evaluation_data():
    """Provides original_df, imputed_df, and missing_mask for evaluation."""
    original_df = pd.DataFrame({
        'Numeric': [1.0, 2.0, 3.0, 4.0, 5.0],
        'Categorical': ['a', 'b', 'a', 'c', 'b'],
        'Numeric_NoMissing': [10, 20, 30, 40, 50]
    })
    
    # Create a missing mask (True where values were missing)
    missing_mask = pd.DataFrame({
        'Numeric': [False, True, False, True, False], # Missing at index 1, 3
        'Categorical': [True, False, False, False, True], # Missing at index 0, 4
        'Numeric_NoMissing': [False, False, False, False, False] # No missing
    })
    
    # Create an imputed DataFrame (can be perfect or imperfect)
    imputed_df = original_df.copy()
    # Simulate imputation (perfect imputation for simplicity here)
    imputed_df.loc[missing_mask['Numeric'], 'Numeric'] = original_df.loc[missing_mask['Numeric'], 'Numeric']
    imputed_df.loc[missing_mask['Categorical'], 'Categorical'] = original_df.loc[missing_mask['Categorical'], 'Categorical']
    # Introduce some errors for realistic testing
    imputed_df.loc[1, 'Numeric'] = 2.5 # Original was 2.0
    imputed_df.loc[3, 'Numeric'] = 3.8 # Original was 4.0
    imputed_df.loc[0, 'Categorical'] = 'b' # Original was 'a'
    imputed_df.loc[4, 'Categorical'] = 'b' # Original was 'b' (correct)

    return original_df, imputed_df, missing_mask

# --- Tests for evaluate_imputation_accuracy ---

def test_evaluate_imputation_accuracy_numeric(evaluation_data):
    """Test evaluation metrics for a numeric column."""
    original_df, imputed_df, missing_mask = evaluation_data
    results = evaluate_imputation_accuracy(original_df, imputed_df, missing_mask)
    
    assert 'Numeric' in results
    numeric_metrics = results['Numeric']
    assert 'mse' in numeric_metrics
    assert 'mae' in numeric_metrics
    assert 'rmse' in numeric_metrics
    assert 'r2' in numeric_metrics
    
    # Expected values based on imputed_df changes:
    # Original: [2.0, 4.0], Imputed: [2.5, 3.8]
    # Errors: [0.5, -0.2]
    # Squared Errors: [0.25, 0.04] -> MSE = (0.25 + 0.04) / 2 = 0.145
    # Absolute Errors: [0.5, 0.2] -> MAE = (0.5 + 0.2) / 2 = 0.35
    # RMSE = sqrt(0.145) approx 0.380788
    np.testing.assert_almost_equal(numeric_metrics['mse'], 0.145)
    np.testing.assert_almost_equal(numeric_metrics['mae'], 0.35)
    np.testing.assert_almost_equal(numeric_metrics['rmse'], np.sqrt(0.145))
    # R2 needs calculation based on variance of original values [2.0, 4.0]
    # Mean = 3.0, Variance = ((2-3)^2 + (4-3)^2) / 2 = (1+1)/2 = 1.0
    # R2 = 1 - (Sum Sq Error / Sum Sq Total) = 1 - (0.25 + 0.04) / ((2-3)^2 + (4-3)^2) = 1 - 0.29 / 2 = 1 - 0.145 = 0.855
    np.testing.assert_almost_equal(numeric_metrics['r2'], 0.855)


def test_evaluate_imputation_accuracy_categorical(evaluation_data):
    """Test evaluation metrics for a categorical column."""
    original_df, imputed_df, missing_mask = evaluation_data
    results = evaluate_imputation_accuracy(original_df, imputed_df, missing_mask)
    
    assert 'Categorical' in results
    categorical_metrics = results['Categorical']
    assert 'accuracy' in categorical_metrics
    
    # Expected values:
    # Original: ['a', 'b'], Imputed: ['b', 'b']
    # Correct: [False, True] -> Accuracy = 1/2 = 0.5
    np.testing.assert_almost_equal(categorical_metrics['accuracy'], 0.5)

def test_evaluate_imputation_accuracy_no_missing(evaluation_data):
    """Test evaluation when a column had no missing values."""
    original_df, imputed_df, missing_mask = evaluation_data
    results = evaluate_imputation_accuracy(original_df, imputed_df, missing_mask)
    
    assert 'Numeric_NoMissing' not in results # Should be skipped

def test_evaluate_imputation_accuracy_perfect_imputation():
    """Test evaluation with perfect imputation."""
    original_df = pd.DataFrame({'A': [1, 2, 3, 4]})
    missing_mask = pd.DataFrame({'A': [False, True, True, False]})
    imputed_df = original_df.copy() # Perfect imputation
    
    results = evaluate_imputation_accuracy(original_df, imputed_df, missing_mask)
    assert 'A' in results
    numeric_metrics = results['A']
    np.testing.assert_almost_equal(numeric_metrics['mse'], 0.0)
    np.testing.assert_almost_equal(numeric_metrics['mae'], 0.0)
    np.testing.assert_almost_equal(numeric_metrics['rmse'], 0.0)
    np.testing.assert_almost_equal(numeric_metrics['r2'], 1.0)

def test_evaluate_imputation_accuracy_empty():
    """Test evaluation with empty inputs."""
    original_df = pd.DataFrame()
    imputed_df = pd.DataFrame()
    missing_mask = pd.DataFrame()
    results = evaluate_imputation_accuracy(original_df, imputed_df, missing_mask)
    assert results == {}


# --- Tests for plot_imputation_evaluation ---

# Use patch to prevent plots from showing during tests
@patch('matplotlib.pyplot.show') 
def test_plot_imputation_evaluation_numeric_runs(mock_show, evaluation_data):
    """Test if plotting runs without error for a numeric column."""
    original_df, imputed_df, missing_mask = evaluation_data
    try:
        plot_imputation_evaluation(original_df, imputed_df, missing_mask, column='Numeric')
        plt.close('all') # Close the figure created by the function
    except Exception as e:
        pytest.fail(f"plot_imputation_evaluation raised an exception for numeric column: {e}")

@patch('matplotlib.pyplot.show')
def test_plot_imputation_evaluation_categorical_runs(mock_show, evaluation_data):
    """Test if plotting runs without error for a categorical column."""
    original_df, imputed_df, missing_mask = evaluation_data
    # Ensure the categorical column has more than one unique value for confusion matrix
    original_df['Categorical'] = ['a', 'b', 'a', 'c', 'b'] 
    imputed_df['Categorical'] = ['b', 'b', 'a', 'c', 'a'] # Example imputed
    missing_mask['Categorical'] = [True, False, True, False, True] # Example mask
    
    try:
        plot_imputation_evaluation(original_df, imputed_df, missing_mask, column='Categorical')
        plt.close('all') # Close the figure
    except Exception as e:
        pytest.fail(f"plot_imputation_evaluation raised an exception for categorical column: {e}")

@patch('matplotlib.pyplot.show')
def test_plot_imputation_evaluation_no_missing_runs(mock_show, evaluation_data, capsys):
    """Test plotting for a column with no missing values (should print message)."""
    original_df, imputed_df, missing_mask = evaluation_data
    try:
        plot_imputation_evaluation(original_df, imputed_df, missing_mask, column='Numeric_NoMissing')
        plt.close('all') # Ensure no figure is left open
        captured = capsys.readouterr()
        assert "No missing values in column 'Numeric_NoMissing'" in captured.out
    except Exception as e:
        pytest.fail(f"plot_imputation_evaluation raised an exception for no-missing column: {e}")
