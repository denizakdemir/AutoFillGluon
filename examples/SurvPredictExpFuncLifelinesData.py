import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter

from lifelines.datasets import (
    load_c_botulinum_lag_phase, load_canadian_senators, load_dd, load_dfcv,
    load_diabetes, load_gbsg2, load_holly_molly_polly, load_kidney_transplant,
    load_larynx, load_lcd, load_leukemia, load_lung, load_lupus, load_lymph_node,
    load_lymphoma, load_mice, load_multicenter_aids_cohort_study, load_nh4,
    load_panel_test, load_psychiatric_patients, load_recur, load_regression_dataset,
    load_rossi, load_stanford_heart_transplants, load_static_test, load_waltons
)


# load each of the datasets, and then print the first few rows, data dimensions, and column names
datasets = [load_c_botulinum_lag_phase, load_canadian_senators, load_dd, load_dfcv,
            load_diabetes, load_gbsg2, load_holly_molly_polly, load_kidney_transplant,
            load_larynx, load_lcd, load_leukemia, load_lung, load_lupus, load_lymph_node,
            load_lymphoma, load_mice, load_multicenter_aids_cohort_study, load_nh4,
            load_panel_test, load_psychiatric_patients, load_recur, load_regression_dataset,
            load_rossi, load_stanford_heart_transplants, load_static_test, load_waltons]

for dataset in datasets:
    data = dataset()
    print(dataset)
    print(data.head())
    print(data.shape)
    print(data.columns)
    print("\n")

# select the datasets with more than 1000 rows and 10 columns
selected_datasets = []

for dataset in datasets:
    data = dataset()
    if data.shape[0] > 500 and data.shape[1] > 7:
        selected_datasets.append(dataset)

# print the names of the selected datasets,and summary statistics for each dataset
for dataset in selected_datasets:
    print(dataset.__name__)
    data = dataset()
    print(data.describe())
    print("\n")

# for load_dd time column is duration, event column is observed
# for load_gbsg2 time column is time, event column is cens
# for load_lymph_node time column is survtime, event column is censdead

# we will only use these three datasets
use_datasets = [load_dd, load_gbsg2, load_lymph_node]



from lifelines.utils import concordance_index
import sys
import json
sys.path.append('scripts')  # Add the directory to Python path

from autogluonImputer import Imputer
from autogluon.tabular import TabularDataset


def run_experiment(data, time_var='time', event_var='status'):
    # Remove unnecessary columns
    for col in data.columns:
        if data[col].nunique() == 1 or data[col].isnull().sum() == data.shape[0] or (data[col].dtype == "object" and data[col].nunique() > 10):
            data.drop(col, axis=1, inplace=True)
    data.dropna(inplace=True)

    # Split data
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

    # Insert missing values
    df_missing_train = X_train.copy()
    for col in df_missing_train.columns:
        #df_missing_train.loc[df_missing_train.sample(frac=0.1).index, col] = np.nan
        0
    for col in df_missing_train.columns:
        if df_missing_train.dtypes[col] == "object":
            df_missing_train[col] = df_missing_train[col].astype("category").replace('nan', np.nan)
        else:
            df_missing_train[col] = df_missing_train[col].astype("float64")
    df_missing_train = TabularDataset(df_missing_train)
    simple_impute_columns=df_missing_train.columns.tolist()
    print(simple_impute_columns)
    # drop duration and event columns
    simple_impute_columns.pop(simple_impute_columns.index(time_var))
    simple_impute_columns.pop(simple_impute_columns.index(event_var))
    print(simple_impute_columns)
    imputer = Imputer(num_iter=2, time_limit=30, presets='medium_quality', column_settings={time_var: {'time_limit': 500, 'presets': 'best_quality'}},simple_impute_columns=simple_impute_columns)
    imputer.fit(df_missing_train)
    # Test data
    df_missing_test = X_test.copy()
    for col in df_missing_test.columns:
        if col != event_var:
            #df_missing_test.loc[df_missing_test.sample(frac=0.2).index, col] = np.nan
            0
    for col in df_missing_test.columns:
        if df_missing_test.dtypes[col] == "object":
            df_missing_test[col] = df_missing_test[col].astype("category").replace('nan', np.nan)
        else:
            df_missing_test[col] = df_missing_test[col].astype("float64")
    df_missing_test[time_var] = np.nan
    # put 1 in event column for all rows
    df_missing_test[event_var] = 1
    df_missing_test = TabularDataset(df_missing_test)
    # Impute test data
    X_test_imputed = imputer.transform(df_missing_test)
    X_train_coded = X_train.copy()
    for col in X_train_coded.columns:
        if X_train_coded[col].dtype == "object":
            le = LabelEncoder()
            X_train_coded[col] = le.fit_transform(X_train_coded[col])

    # Cox regression
    cph = CoxPHFitter()

    cph.fit(X_train_coded, duration_col=time_var, event_col=event_var)
    # Fit the WeibullAFTFitter
    aft = WeibullAFTFitter()
    aft.fit(X_train_coded, duration_col=time_var, event_col=event_var)

    # Predict
    X_test_coded = X_test.copy()

    for col in X_test_coded.columns:
        if X_test_coded[col].dtype == "category":
            le = LabelEncoder()
            X_test_coded[col] = le.fit_transform(X_test_coded[col])

    # Convert types for columns with mismatched dtypes in X_train_coded and X_test_coded
    for col in X_test_coded.columns:
        if X_test_coded[col].dtype != X_train_coded[col].dtype:
            if X_test_coded[col].dtype == "object":
                le = LabelEncoder()
                X_test_coded[col] = le.fit_transform(X_test_coded[col])
            else:
                X_test_coded[col] = X_test_coded[col].astype("float64")

    predtest = cph.predict_expectation(X_test_coded)

    c_index_imputed = concordance_index(X_test_coded[time_var], X_test_imputed[time_var], X_test_coded[event_var])
    c_index_pred = concordance_index(X_test_coded[time_var], predtest, X_test_coded[event_var])
    # Predict using the WeibullAFTFitter
    mean_pred_survival = aft.predict_expectation(X_test_coded)
    # Evaluate
    c_index_weibull = concordance_index(X_test_coded[time_var], mean_pred_survival, X_test_coded[event_var])
    return c_index_imputed, c_index_pred, c_index_weibull

##################################################
# Define the time and event columns for each dataset
dataset_columns = {
    "load_dd": {"time": "duration", "event": "observed"},
    "load_gbsg2": {"time": "time", "event": "cens"},
    "load_lymph_node": {"time": "survtime", "event": "censdead"}
}

# Dictionary to store results for each dataset
results = {}

# Loop through the datasets and run the experiment
for dataset in use_datasets:
    dataset_name = dataset.__name__
    print(f"Processing {dataset_name}...");
    data = dataset()
    if dataset_name == "load_lymph_node":
        # drop columns 'rectime', 'censrec', recdate   deathdate
        data.drop(['rectime', 'censrec','recdate', 'deathdate','id','diagdateb'], axis=1, inplace=True)
    c_index_imputed, c_index_pred, c_index_weibull = run_experiment(data, time_var=dataset_columns[dataset_name]["time"], event_var=dataset_columns[dataset_name]["event"])
    # Store the results
    results[dataset_name] = {
        "C-index for imputed data": c_index_imputed,
        "C-index for predictions": c_index_pred,
        "C-index for Weibull": c_index_weibull
    }
    # Print the results for the current dataset
    print("C-index for imputed data:", c_index_imputed)
    print("C-index for predictions:", c_index_pred)
    print("C-index for Weibull:", c_index_weibull)
    print("\n")

# Print results for all datasets
for dataset, dataset_results in results.items():
    print(f"Results for {dataset}:")
    for metric, value in dataset_results.items():
        print(metric, ":", round(value, 3))
    print("\n")
