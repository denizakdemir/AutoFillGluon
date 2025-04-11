import pytest
import pandas as pd
import numpy as np
import os
import shutil
from autogluon.tabular import TabularDataset
from autofillgluon.imputer.imputer import Imputer, multiple_imputation # Adjusted import

# --- Fixtures ---

@pytest.fixture(scope="module")
def sample_df():
    """Create a larger base DataFrame (15 rows) for more robust testing."""
    data = {
        'num1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        'num2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        'cat1': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'], # Ensure all 3 classes are present multiple times
    }
    df = pd.DataFrame(data)
    df['cat1'] = df['cat1'].astype('category')
    return TabularDataset(df)

@pytest.fixture
def sample_df_missing(sample_df):
    """Create a larger DataFrame (15 rows) with missing values."""
    df_missing = sample_df.copy()
    # Introduce missing values at different positions, keeping enough non-NaN for cat1
    df_missing.loc[[0, 5, 10], 'num1'] = np.nan # 3 NaNs
    df_missing.loc[[1, 6, 11], 'num2'] = np.nan # 3 NaNs
    df_missing.loc[[2, 7], 'cat1'] = np.nan    # 2 NaNs (leaves 13 non-NaN for cat1)
    return df_missing

@pytest.fixture
def imputer_instance():
    """Basic Imputer instance for testing."""
    # Use low num_iter and time_limit for faster tests
    # Use optimize_for_deployment preset which is faster and more stable on M1
    return Imputer(
        num_iter=1,
        time_limit=30,
        presets=['optimize_for_deployment'],
        use_missingness_features=False
    )

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup fixture that runs after each test."""
    yield
    # Clean up any AutogluonModels directory
    if os.path.exists('AutogluonModels'):
        shutil.rmtree('AutogluonModels')
    # Clean up any temporary model directories
    for item in os.listdir('.'):
        if item.startswith('ag_imputer_') and os.path.isdir(item):
            shutil.rmtree(item)

# --- Basic Tests (Refactored from unittest) ---

def test_imputer_initialization():
    """Test that the Imputer can be initialized."""
    imputer = Imputer(num_iter=1, time_limit=5)
    assert isinstance(imputer, Imputer)
    assert imputer.num_iter == 1
    assert imputer.time_limit == 5# @pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit") # Keep xfail commented out if skip is used
def test_imputer_fit(imputer_instance, sample_df_missing):
    """Test that the Imputer can fit a dataframe with missing values."""
    df_imputed = imputer_instance.fit(sample_df_missing)
    
    # Check that the returned dataframe has no missing values
    assert not df_imputed.isnull().any().any()
    # Check internal state after fit
    assert len(imputer_instance.models) > 0 # Should have models for cols with NaNs
    assert 'num1' in imputer_instance.models
    assert 'num2' in imputer_instance.models
    assert 'cat1' in imputer_instance.models
    assert len(imputer_instance.initial_imputes) == len(sample_df_missing.columns)
    assert len(imputer_instance.colsummary) == len(sample_df_missing.columns)
    assert imputer_instance.missing_cells is not None

    # Check model directory structure (assuming default 'AutogluonModels')
    model_base_path = "AutogluonModels"
    if os.path.exists(model_base_path):
        # Check for expected predictor directories
        assert os.path.isdir(os.path.join(model_base_path, "ag_imputer_num1"))
        assert os.path.isdir(os.path.join(model_base_path, "ag_imputer_num2"))
        assert os.path.isdir(os.path.join(model_base_path, "ag_imputer_cat1"))
        # Check that *no* timestamped directories exist directly under the base path
        # (This assumes the fit process doesn't create other unrelated timestamped dirs)
        # Note: This check might be fragile if other processes create such dirs.
        # A more robust check might involve listing dirs before/after or controlling the path.
        # For now, we check common patterns.
        # timestamp_pattern = re.compile(r"ag-\d{8}_\d{6}") # Example pattern
        # assert not any(timestamp_pattern.match(d) for d in os.listdir(model_base_path) if os.path.isdir(os.path.join(model_base_path, d)))

        # Clean up after check
        shutil.rmtree(model_base_path)
    else:
        # If the base path wasn't created (e.g., no models trained?), the test might need adjustment
        # For this test case, we expect models, so the path should exist.
        pytest.fail(f"Expected model directory '{model_base_path}' not found after fit.")

# @pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_imputer_transform(imputer_instance, sample_df, sample_df_missing):
    """Test that the Imputer can transform a new dataframe."""
    imputer_instance.fit(sample_df_missing) # Fit first
    
    # Create a new dataframe with different missing values
    new_df_missing = sample_df.copy()
    new_df_missing.loc[3, 'num1'] = np.nan
    new_df_missing.loc[4, 'cat1'] = np.nan
    
    # Transform the new dataframe
    new_df_imputed = imputer_instance.transform(new_df_missing)
    
    # Check that the returned dataframe has no missing values
    assert not new_df_imputed.isnull().any().any()# @pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_column_types_preserved(imputer_instance, sample_df_missing):
    """Test that the Imputer preserves column types after fit."""
    df_imputed = imputer_instance.fit(sample_df_missing)
    
    # Check that numerical columns are still numerical
    assert np.issubdtype(df_imputed['num1'].dtype, np.number)
    assert np.issubdtype(df_imputed['num2'].dtype, np.number)
    
    # Check that categorical columns are still categorical
    assert pd.api.types.is_categorical_dtype(df_imputed['cat1'].dtype)
# --- Tests for Save/Load Functionality ---

# @pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_save_load_models(imputer_instance, sample_df_missing, tmp_path):
    """Test saving and loading the imputer state."""
    # Fit the imputer
    imputer_instance.fit(sample_df_missing)
    
    # Get state before saving
    original_models_keys = set(imputer_instance.models.keys())
    original_initial_imputes = imputer_instance.initial_imputes.copy()
    original_colsummary = imputer_instance.colsummary.copy()
    
    # Define save path
    save_dir = tmp_path / "test_imputer_save"
    
    # Save models
    imputer_instance.save_models(str(save_dir)) # save_models expects string path
    
    # Debug: Print directory contents
    print(f"\nDirectory contents of {save_dir}:")
    if os.path.exists(save_dir):
        for item in os.listdir(save_dir):
            print(f"  - {item}")
    else:
        print("  Directory does not exist!")
    
    # Check if files/dirs were created
    assert os.path.exists(save_dir), f"Save directory {save_dir} does not exist"
    
    # Check for model directories
    for col in ['num1', 'cat1']:
        model_dir = save_dir / f'ag_imputer_{col}'
        assert os.path.exists(model_dir), f"Model directory {model_dir} does not exist"
    
    # Check for metadata files
    assert os.path.exists(save_dir / 'initial_imputes.pkl'), "initial_imputes.pkl not found"
    assert os.path.exists(save_dir / 'colsummary.pkl'), "colsummary.pkl not found"
    assert os.path.exists(save_dir / 'model_columns.pkl'), "model_columns.pkl not found"
    
    # Create a new imputer instance
    new_imputer = Imputer()
    
    # Load models
    new_imputer.load_models(str(save_dir))
    
    # Check if the state is restored
    assert set(new_imputer.models.keys()) == original_models_keys

    # Compare initial_imputes dictionaries
    assert set(new_imputer.initial_imputes.keys()) == set(original_initial_imputes.keys())
    for col in new_imputer.initial_imputes:
        pd.testing.assert_series_equal(new_imputer.initial_imputes[col], original_initial_imputes[col])

    # Compare colsummary dictionaries
    assert new_imputer.colsummary == original_colsummary

    # Test if the loaded imputer can transform data
    transformed_df = new_imputer.transform(sample_df_missing.copy())
    assert not transformed_df.isnull().any().any()
    
    # Clean up temporary AutogluonModels directory if created by fit/transform
    # Clean up temporary AutogluonModels directory if created by fit/transform
    # It's better practice to let pytest handle cleanup or use specific paths
    # within tmp_path for model saving during fit if possible.
    # For now, keep the manual cleanup.
    if os.path.exists('AutogluonModels'):
        shutil.rmtree('AutogluonModels')

# --- Tests for Imputer Options ---

@pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_imputer_fit_with_missingness_features(sample_df_missing):
    """Test fitting with use_missingness_features=True."""
    imputer = Imputer(num_iter=1, time_limit=10, use_missingness_features=True)
    df_imputed = imputer.fit(sample_df_missing)

    # Check if missingness columns were added to the internal representation used for training
    # The returned df_imputed should NOT have the _missing columns
    assert not any(col.endswith('_missing') for col in df_imputed.columns)
    assert not df_imputed.isnull().any().any()

    # Check internal state: models should potentially use these features
    # We can check if the features used by a model include the _missing columns
    # This requires inspecting the predictor object more deeply
    predictor_num1 = imputer.models.get('num1')
    if predictor_num1:
        # Note: Accessing internal predictor features might be fragile
        try:
             # Get features used by the predictor (may vary based on AutoGluon version/internals)
             # This is an example, the actual way to get features might differ
             features_used = predictor_num1.feature_metadata_in.get_features()
             assert 'num2_missing' in features_used
             assert 'cat1_missing' in features_used
        except AttributeError:
             pytest.skip("Could not access predictor features for verification (internal API might have changed).")

    # Check that initial imputes and colsummary do not include _missing cols
    assert not any(col.endswith('_missing') for col in imputer.initial_imputes)
    assert not any(col.endswith('_missing') for col in imputer.colsummary)
@pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_imputer_transform_with_missingness_features(sample_df, sample_df_missing):
    """Test transform with use_missingness_features=True."""
    imputer = Imputer(num_iter=1, time_limit=10, use_missingness_features=True)
    imputer.fit(sample_df_missing) # Fit first

    # Create a new dataframe with different missing values
    new_df_missing = sample_df.copy()
    new_df_missing.loc[3, 'num1'] = np.nan
    new_df_missing.loc[4, 'cat1'] = np.nan

    # Transform the new dataframe
    new_df_imputed = imputer.transform(new_df_missing)

    # Check that the returned dataframe has no missing values and no _missing columns
    assert not new_df_imputed.isnull().any().any()
    assert not any(col.endswith('_missing') for col in new_df_imputed.columns)

    # Clean up temporary AutogluonModels directory
    if os.path.exists('AutogluonModels'):
        shutil.rmtree('AutogluonModels')
@pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_imputer_with_simple_impute_columns(sample_df_missing):
    """Test fitting and transforming with simple_impute_columns."""
    simple_cols = ['num1'] # Choose a column with missing values
    imputer = Imputer(num_iter=1, time_limit=10, simple_impute_columns=simple_cols)
    
    # --- Test Fit ---
    df_imputed_fit = imputer.fit(sample_df_missing.copy()) # Use copy to avoid modifying fixture
    
    # Check no missing values
    assert not df_imputed_fit.isnull().any().any()
    
    # Check models: 'num1' should NOT have a model, others should
    assert simple_cols[0] not in imputer.models
    assert 'num2' in imputer.models
    assert 'cat1' in imputer.models
    
    # Check if the value imputed for num1 (at index 0) is the initial mean
    initial_mean_num1 = imputer.initial_imputes['num1']
    assert df_imputed_fit.loc[0, 'num1'] == initial_mean_num1

    # --- Test Transform ---
    # Create new missing data, including in the simple column
    new_df_missing = sample_df_missing.copy()
    new_df_missing.loc[3, 'num1'] = np.nan # Add another NaN in simple col
    new_df_missing.loc[4, 'num2'] = np.nan # Add NaN in model col
    
    df_imputed_transform = imputer.transform(new_df_missing)
    
    # Check no missing values
    assert not df_imputed_transform.isnull().any().any()
    
    # Check if the newly added NaN in num1 (index 3) was imputed with the initial mean
    assert df_imputed_transform.loc[3, 'num1'] == initial_mean_num1
    # Check if the newly added NaN in num2 (index 4) was imputed (value depends on model)
    assert pd.notna(df_imputed_transform.loc[4, 'num2'])

    # Clean up temporary AutogluonModels directory
    if os.path.exists('AutogluonModels'):
        shutil.rmtree('AutogluonModels')
# @pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_imputer_with_column_settings(sample_df_missing):
    """Test fitting with column_settings parameter."""
    col_settings = {
        'num1': {'time_limit': 5, 'presets': ['good_quality']}, # Different settings for num1
        'cat1': {'eval_metric': 'accuracy'} # Specify eval_metric for categorical
    }
    imputer = Imputer(num_iter=1, time_limit=10, column_settings=col_settings) # Default time is 10

    try:
        df_imputed = imputer.fit(sample_df_missing.copy())
        # Primary check: Did it run without errors?
        assert not df_imputed.isnull().any().any()
        # Check if models were created for columns with settings
        assert 'num1' in imputer.models
        assert 'cat1' in imputer.models
        # Optional: Check if eval_metric was set (might be hard to verify directly)
        # predictor_cat1 = imputer.models.get('cat1')
        # if predictor_cat1:
        #     assert predictor_cat1.eval_metric_name == 'accuracy' # Accessing internal attributes can be brittle
    except Exception as e:
        pytest.fail(f"Imputer fit failed with column_settings: {e}")
    finally:
        # Clean up temporary AutogluonModels directory
        if os.path.exists('AutogluonModels'):
            shutil.rmtree('AutogluonModels')

# --- Tests for Other Imputer Methods ---

# @pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_feature_importance(imputer_instance, sample_df_missing):
    """Test the feature_importance method."""
    df_imputed = imputer_instance.fit(sample_df_missing.copy())
    
    try:
        importances = imputer_instance.feature_importance(df_imputed)
        
        # Check return type
        assert isinstance(importances, dict)
        
        # Check keys match modeled columns
        modeled_cols = set(imputer_instance.models.keys())
        assert set(importances.keys()) == modeled_cols
        
        # Check value types (should be DataFrame or None)
        for col, importance_df in importances.items():
             # Allow for None if model doesn't support feature importance or error occurred
            assert importance_df is None or isinstance(importance_df, pd.DataFrame)
            if isinstance(importance_df, pd.DataFrame):
                assert not importance_df.empty
                assert 'importance' in importance_df.columns # Common column name

    except Exception as e:
        pytest.fail(f"feature_importance raised an exception: {e}")
    finally:
        # Clean up temporary AutogluonModels directory
        if os.path.exists('AutogluonModels'):
            shutil.rmtree('AutogluonModels')
# @pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_add_missingness_at_random(imputer_instance, sample_df):
    """Test the add_missingness_at_random method."""
    # Fit the imputer first to populate self.models (needed by the method)
    # Create a dummy missing df just for fitting
    dummy_missing = sample_df.copy()
    dummy_missing.loc[0, 'num1'] = np.nan 
    imputer_instance.fit(dummy_missing)

    original_data = sample_df.copy()
    percentage = 0.2 # Add 20% missingness

    modified_data, missingness_indices = imputer_instance.add_missingness_at_random(original_data, percentage)

    # Check return types
    assert isinstance(modified_data, pd.DataFrame)
    assert isinstance(missingness_indices, dict)

    # Check that the original dataframe is not modified
    assert original_data.equals(sample_df) 
    # Check that the modified dataframe is different
    assert not modified_data.equals(original_data)

    total_added_missing = 0
    for col, indices in missingness_indices.items():
        # Check indices are valid
        assert all(idx in original_data.index for idx in indices)
        # Check that the values at these indices in modified_data are NaN
        assert modified_data.loc[indices, col].isnull().all()
        # Check that the values at these indices in original_data were NOT NaN
        assert original_data.loc[indices, col].notnull().all()
        
        # Calculate expected number of missing values for this column
        non_missing_count = original_data[col].notna().sum()
        expected_missing_count = int(non_missing_count * percentage)
        # Allow for slight rounding differences
        assert abs(len(indices) - expected_missing_count) <= 1 
        total_added_missing += len(indices)

    # Check overall percentage (approximate due to per-column calculation)
    total_non_missing = original_data.notna().sum().sum()
    expected_total_missing = int(total_non_missing * percentage)
    # This check is less precise because it's summed across columns
    assert abs(total_added_missing - expected_total_missing) <= len(original_data.columns) 

    # Clean up temporary AutogluonModels directory
    if os.path.exists('AutogluonModels'):
        shutil.rmtree('AutogluonModels')
# @pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_evaluate_imputation(imputer_instance, sample_df):
    """Test the evaluate_imputation method."""
    # Fit the imputer first
    dummy_missing = sample_df.copy()
    dummy_missing.loc[0, 'num1'] = np.nan 
    dummy_missing.loc[1, 'num2'] = np.nan
    dummy_missing.loc[2, 'cat1'] = np.nan
    imputer_instance.fit(dummy_missing)

    original_data = sample_df.copy()
    percentage = 0.2 # Evaluate on 20% missingness
    ntimes = 2 # Run evaluation twice for testing

    try:
        evaluation_results = imputer_instance.evaluate_imputation(original_data, percentage, ntimes=ntimes)

        # Check return type and structure
        assert isinstance(evaluation_results, dict)
        assert len(evaluation_results) == ntimes # Check number of repetitions
        assert all(isinstance(res, dict) for res in evaluation_results.values())

        # Check content of one repetition
        rep_result = evaluation_results[0] # Check the first repetition
        modeled_cols = set(imputer_instance.models.keys())
        assert set(rep_result.keys()) == modeled_cols # Should have results for modeled columns

        # Check metrics for a numeric column
        if 'num1' in rep_result:
            assert 'mse' in rep_result['num1']
            assert 'mae' in rep_result['num1']
            assert isinstance(rep_result['num1']['mse'], (int, float))
            assert isinstance(rep_result['num1']['mae'], (int, float))

        # Check metrics for a categorical column
        if 'cat1' in rep_result:
            assert 'accuracy' in rep_result['cat1']
            assert isinstance(rep_result['cat1']['accuracy'], (int, float))
            assert 0 <= rep_result['cat1']['accuracy'] <= 1

    except Exception as e:
        pytest.fail(f"evaluate_imputation raised an exception: {e}")
    finally:
        # Clean up temporary AutogluonModels directory
        if os.path.exists('AutogluonModels'):
            shutil.rmtree('AutogluonModels')

# --- Tests for multiple_imputation function ---

# @pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_multiple_imputation_fit_each_time(sample_df_missing):
    """Test multiple_imputation with fitonce=False."""
    n_imputations = 2
    try:
        imputed_datasets = multiple_imputation(
            sample_df_missing.copy(), 
            n_imputations=n_imputations, 
            fitonce=False, 
            num_iter=1, 
            time_limit=10 # Keep tests fast
        )
        
        # Check return type and length
        assert isinstance(imputed_datasets, list)
        assert len(imputed_datasets) == n_imputations
        
        # Check each dataset
        for df_imputed in imputed_datasets:
            assert isinstance(df_imputed, pd.DataFrame)
            assert not df_imputed.isnull().any().any()
            # Check columns match original
            assert list(df_imputed.columns) == list(sample_df_missing.columns)
            
    except Exception as e:
        pytest.fail(f"multiple_imputation (fitonce=False) raised an exception: {e}")
    finally:
        # Clean up multiple AutogluonModels directories if created
        # This cleanup might be complex if paths aren't controlled.
        # A better approach might involve passing a base temp path to Imputer.
        if os.path.exists('AutogluonModels'):
             # Basic cleanup, might leave nested dirs if fit creates unique paths
            shutil.rmtree('AutogluonModels', ignore_errors=True)
            # Need more robust cleanup if fit creates timestamped dirs outside AutogluonModels
# @pytest.mark.xfail(reason="Segmentation fault during AutoGluon fit")
def test_multiple_imputation_fit_once(sample_df_missing):
    """Test multiple_imputation with fitonce=True."""
    n_imputations = 2
    try:
        imputed_datasets = multiple_imputation(
            sample_df_missing.copy(), 
            n_imputations=n_imputations, 
            fitonce=True, 
            num_iter=1, 
            time_limit=10 # Keep tests fast
        )
        
        # Check return type and length
        assert isinstance(imputed_datasets, list)
        assert len(imputed_datasets) == n_imputations
        
        # Check each dataset
        for df_imputed in imputed_datasets:
            assert isinstance(df_imputed, pd.DataFrame)
            assert not df_imputed.isnull().any().any()
            # Check columns match original
            assert list(df_imputed.columns) == list(sample_df_missing.columns)

    except Exception as e:
        pytest.fail(f"multiple_imputation (fitonce=True) raised an exception: {e}")
    finally:
        # Clean up temporary AutogluonModels directory
        if os.path.exists('AutogluonModels'):
            shutil.rmtree('AutogluonModels')

@pytest.mark.filterwarnings("ignore:.*auto.dataset_overview.*:DeprecationWarning") # Ignore potential EDA deprecation warnings
def test_dataset_overview_runs(imputer_instance, sample_df, capsys):
    """Smoke test for the dataset_overview method."""
    # Requires train_data, test_data, label
    train_data = sample_df.copy()
    test_data = sample_df.copy()
    label = 'cat1' # Choose one column as the label for the test

    try: # Corrected indentation
        # Simply call the function. The test passes if no exception is raised.
        imputer_instance.dataset_overview(train_data=train_data, test_data=test_data, label=label)
        # We capture output just in case, but don't assert on it for a basic smoke test.
        _ = capsys.readouterr()
    except Exception as e:
        # Fail if any unhandled exception occurs during the call
        pytest.fail(f"dataset_overview raised an exception: {e}")
    finally:
        # Clean up any potential side effects if EDA creates files/dirs
        pass # Corrected indentation - EDA overview usually doesn't create persistent files in CWD


# TODO: Add test for redirect_stdout_to_file context manager if needed explicitly
