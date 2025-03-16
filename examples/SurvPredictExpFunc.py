import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter

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
        df_missing_train.loc[df_missing_train.sample(frac=0.1).index, col] = np.nan
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
            df_missing_test.loc[df_missing_test.sample(frac=0.2).index, col] = np.nan
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
    c_index_weibull = concordance_index(X_test_coded[time_var], mean_pred_survival, X_test_coded[event_var]) # we use negative since we want to rank higher times as better

    return c_index_imputed, c_index_pred, c_index_weibull

##################################################
data = pd.read_csv('SurvivalDatasets/Lung_data.csv', index_col=0)
                     # replace with the path to your DataFrame
# Run experiment 10 times
c_indices_imputed = []
c_indices_pred = []
c_indices_weibull = []

for i in range(10):
    c_index_imputed, c_index_pred, c_index_weibull= run_experiment(data)
    c_indices_imputed.append(c_index_imputed)
    c_indices_pred.append(c_index_pred)
    c_indices_weibull.append(c_index_weibull)
    # print all the c-indices
    print("C-index for imputed data:", c_index_imputed)
    print("C-index for predictions:", c_index_pred)
    print("C-index for Weibull:", c_index_weibull)



print("C-indices for imputed data:", c_indices_imputed)
print("C-indices for predictions:", c_indices_pred)
print("C-indices for Weibull:", c_indices_weibull)

print("Mean C-index for imputed data:", np.mean(c_indices_imputed))
print("Mean C-index for predictions:", np.mean(c_indices_pred))
print("Mean C-index for Weibull:", np.mean(c_indices_weibull))

######################################################################
import os

# Directory containing your datasets
DATA_DIR = 'SurvivalDatasets'

# if the results folder does not exist, create it
if not os.path.exists('Results'):
    os.makedirs('Results')

resultssavepath='Results/results.json'

# List all the .csv files in the directory
all_datasets = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

# Dictionary to store results for each dataset
results = {}

for dataset_file in all_datasets:
    # if p>n, skip the dataset
    data_path = os.path.join(DATA_DIR, dataset_file)
    data = pd.read_csv(data_path, index_col=0)
    if data.shape[1]*5 > data.shape[0]:
        continue
    print(f"Processing {dataset_file}...")
    
    data_path = os.path.join(DATA_DIR, dataset_file)
    data = pd.read_csv(data_path, index_col=0)

    # Run experiment 10 times for each dataset
    c_indices_imputed = []
    c_indices_pred = []
    c_indices_weibull = []

    for i in range(10):
        c_index_imputed, c_index_pred, c_index_weibull = run_experiment(data)
        c_indices_imputed.append(c_index_imputed)
        c_indices_pred.append(c_index_pred)
        c_indices_weibull.append(c_index_weibull)
        # print all the c-indices
        print("C-index for imputed data:", c_index_imputed)
        print("C-index for predictions:", c_index_pred)
        print("C-index for Weibull:", c_index_weibull)

    dataset_results = {
        "C-indices for imputed data": c_indices_imputed,
        "C-indices for predictions": c_indices_pred,
        "C-indices for Weibull": c_indices_weibull,
        "Mean C-index for imputed data": np.mean(c_indices_imputed),
        "Mean C-index for predictions": np.mean(c_indices_pred),
        "Mean C-index for Weibull": np.mean(c_indices_weibull)
    }

    results[dataset_file] = dataset_results
    # Save results after each dataset, overwriting the previous results
    with open(resultssavepath, 'w') as f:
        json.dump(results, f)

# Print results for all datasets
for dataset, dataset_results in results.items():
    print(f"Results for {dataset}:")
    for metric, value in dataset_results.items():
        if "C-indices" in metric:
            print(metric, ":", value)
        else:
            print(metric, ":", round(value, 3))
    print("\n")
