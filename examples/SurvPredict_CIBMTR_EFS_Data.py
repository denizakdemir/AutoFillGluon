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
from autogluon.tabular import TabularDataset

from autogluonImputer import Imputer
from  lifelines.metrics import concordance_index
from autogluon.core.metrics import make_scorer

from autogluon.tabular import TabularDataset

from scorer.scorer import negative_log_likelihood_exponential
from scorer.scorer import scorefunct_coxPH



negative_log_likelihood_exponential_custom_scorer = make_scorer(name='negative_log_likelihood_exponential',
                                                    score_func=negative_log_likelihood_exponential, 
                                                    optimum=-np.inf,
                                                    greater_is_better=False)

scorefunct_coxPH_custom_scorer = make_scorer(name='scorefunct_coxPH',
                                                    score_func=scorefunct_coxPH, 
                                                    optimum=-np.inf,
                                                    greater_is_better=False)

# test the coxPH_custom_scorer
y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y_pred = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4])
print(negative_log_likelihood_exponential_custom_scorer(y_true, y_pred))

# import /Users/dakdemir/Dropbox/AutoGluonImputer/CIBMTR_Data_Brent/analysisdatasetv5.sas7bdat  sas data as a data frame


import pyreadstat

df, meta = pyreadstat.read_sas7bdat('CIBMTR_Data_Brent/analysisdatasetv5.sas7bdat')

# done! let's see what we got
print(df.head())
print(meta.column_names)
print(meta.column_labels)
print(meta.column_names_to_labels)
print(meta.number_rows)
print(meta.number_columns)
print(meta.file_label)
print(meta.file_encoding)
# there are other metadata pieces extracted. See the documentation for more details.


#######


def processdf(df):
    df.dtypes
    df['dead'].value_counts()
    data=df.copy()
    time_var='intxsurv'
    event_var='dead'
    # remove columns with all missing values
    for col in data.columns:
        if data[col].nunique() == 1 or data[col].isnull().sum() == data.shape[0] or (data[col].dtype == "object" and data[col].nunique() > 10):
            data.drop(col, axis=1, inplace=True)

    # remove columns with all same values
    for col in data.columns:
        if data[col].nunique() == 1:
            data.drop(col, axis=1, inplace=True)

    # remove rows with all missing values
    data.dropna(axis=0, how='all', inplace=True)
    data.dtypes
    # remove numeric variables that are actually categorical id's
    # we can check this by checking the whether the variable is coded as integer, floor(x)==x, and the number of categories is more than 100.
    for col in data.columns:
        if data[col].dtype == "float64" and data[col].apply(lambda x: x.is_integer()).all() and data[col].nunique() > 100:
            data.drop(col, axis=1, inplace=True)
    data.dtypes

    # for a column, if 99 is missing (np.nan) if 100 or 98 are not  valud values for that column
    for col in data.columns:
        if (data[col].dtype == "float64" and  len(set(data[col].unique()).intersection(set([99.0, 100.0]))) == 0):
            data[col] = data[col].replace(99.0, np.nan)

    # for a column, if 999 is missing (np.nan) if 1000 is not a valud value for that column
    for col in data.columns:
        if (data[col].dtype == "float64" and len(set(data[col].unique()).intersection(set([998.0, 1000.0]))) == 0):
            data[col] = data[col].replace(999.0, np.nan)
    # remove any rows that have a missing value for either of the time or event variables
    data.dropna(axis=0, subset=[time_var, event_var], inplace=True)
    data.dtypes





    # show a summary of each column
    data.describe(include="all")

    # which variables have value of 99?
    for col in data.columns:
        if data[col].dtype == "float64" and 99.0 in data[col].unique():
            print(col, data[col].unique())

    # all the above 99's are missing values, so we can replace them with np.nan
    for col in data.columns:
        if data[col].dtype == "float64" and 99.0 in data[col].unique():
            data[col] = data[col].replace(99.0, np.nan)

    # show a summary of each column
    data.describe(include="all")

    # print the number of missing values in each column
    data.isnull().sum()
    # reorder to have survival time and event at the beginning
    data = data[[time_var, event_var] + [col for col in data.columns if col not in [time_var, event_var]]]


    # remove any variable that has name start with 'intx', then also remove the variables that has name that is equal to the name of the intx variable intx removed.
    intxcols=[]
    intxcorrespondingcols=[]
    for col in data.columns:
        if col.startswith("intx"):
            intxcols.append(col)
            intxcorrespondingcols.append(col[4:])



    # drop intxsurv from intxcols:
    intxcols.pop(intxcols.index('intxsurv'))
    intxcorrespondingcols.pop(intxcorrespondingcols.index('surv'))

    # drop the corresponding columns and intx columns
    data.drop(intxcols+intxcorrespondingcols, axis=1, inplace=True)

    # describe the data
    data.describe(include="all")

    # lets see the last 10 columns
    data.iloc[:, -3:].describe(include="all")

    # drop any variable which has more than 30% missing values
    data.drop([col for col in data.columns if data[col].isnull().sum() > 0.3 * data.shape[0]], axis=1, inplace=True)

    # describe the data
    data.describe(include="all")

    # drop the last 2 columns
    data.drop(data.columns[-2:], axis=1, inplace=True)

    # describe the data
    data.describe(include="all")

    # print column names
    data.columns.tolist()

    # for each categorical variable, print the table for categories, including the missing values
    for col in data.columns:
        if data[col].dtype == "object":
            print(data[col].value_counts(dropna=False))

    # remove any variable that starts with pseudo
    data.drop([col for col in data.columns if col.startswith("pseudo")], axis=1, inplace=True)

    # table for txnum variable
    data["txnum"].value_counts(dropna=False)

    # column names
    data.columns.tolist()

    # remove  'fult12', 'efsfult12' if they are in the data
    if 'fult12' in data.columns:
        data.drop(['fult12'], axis=1, inplace=True)
    if 'efsfult12' in data.columns:
        data.drop(['efsfult12'], axis=1, inplace=True)

    data.columns.tolist()
    return data


data=processdf(df)

data.dtypes
data.shape
data.columns.tolist()


# drop 'randomuni', 'training'
data.drop(['randomuni', 'training'], axis=1, inplace=True)
data.head()

# print the number of missing values in each column
data.isnull().sum()
data.shape











def run_experiment(data, time_var='intxsurv', event_var='dead', expvars=[]):
    # Remove unnecessary columns
    for col in data.columns:
        if data[col].nunique() == 1 or data[col].isnull().sum() == data.shape[0] or (data[col].dtype == "object" and data[col].nunique() > 10):
            data.drop(col, axis=1, inplace=True)

    # Split data based on trainin column
    X_train, X_test = train_test_split(data, test_size=0.2)
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
    simple_impute_columns=df_missing_train.columns.tolist()
    print(simple_impute_columns)
    # drop duration and event columns
    simple_impute_columns.pop(simple_impute_columns.index(time_var))
    simple_impute_columns.pop(simple_impute_columns.index(event_var))
    print(simple_impute_columns)
    imputer = Imputer(num_iter=1, time_limit=30, presets='medium_quality', column_settings={'time': {'time_limit': 500, 'presets': 'best_quality', 'eval_metric': scorefunct_coxPH_custom_scorer}},simple_impute_columns=simple_impute_columns)
    df_missing_train.dtypes.to_list()
    # reindex
    df_missing_train.reset_index(drop=True, inplace=True)
    # add a column time that is the the week column with negative values if censored and positive if not
    df_missing_train['time'] = df_missing_train[time_var]
    df_missing_train.loc[df_missing_train[event_var] == 0,'time' ] = -df_missing_train[time_var]

    df_missing_train=df_missing_train.drop(columns=[time_var, event_var])
    df_missing_train.reset_index(drop=True, inplace=True)
    df_missing_train = TabularDataset(df_missing_train)
    df_missing_train.reset_index(drop=True, inplace=True)
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

    df_missing_test=df_missing_test.drop(columns=[time_var, event_var])
    # add time column, all values are nan
    df_missing_test['time'] = np.nan
    # make data type of df_missing_test['time'] the same as df_missing_train['time']
    df_missing_test['time'] = df_missing_test['time'].astype(df_missing_train['time'].dtype)



    df_missing_test = TabularDataset(df_missing_test)
    # Impute test data
    X_test_imputed = imputer.transform(df_missing_test)


    X_train_coded = X_train.copy()
    for col in X_train_coded.columns:
        if X_train_coded[col].dtype == "object":
            le = LabelEncoder()
            X_train_coded[col] = le.fit_transform(X_train_coded[col])

    # do a mean imputation for the missing values in the training data for numeric variables
    # and replace the missing values in the categorical variables with the most frequent value
    for col in X_train_coded.columns:
        if X_train_coded[col].dtype == "numeric":
            X_train_coded[col] = X_train_coded[col].fillna(X_train_coded[col].mean())
        else:
            X_train_coded[col] = X_train_coded[col].fillna(X_train_coded[col].mode()[0])
    # number of Nan or na values in each column
    X_train_coded.isnull().sum()
    X_train_coded.dtypes
    X_train_coded.shape

    
    from sklearn.decomposition import PCA
    # calculathe the principal components for columns other than the time and event columns
    X_train_coded_pcs=PCA(n_components=10).fit_transform(X_train_coded.drop([time_var, event_var], axis=1))
    X_train_coded_pcs.shape
    # convert the principal components to a data frame
    X_train_coded_pcs = pd.DataFrame(X_train_coded_pcs, columns=[f"PC{i}" for i in range(1, 11)])
    #add time and event columns to the principal components
    outcomedf=X_train_coded[[time_var, event_var]]
    # drop index
    outcomedf.reset_index(drop=True, inplace=True)
    X_train_coded_pcs.reset_index(drop=True, inplace=True)
    X_train_coded_pcs=pd.concat([X_train_coded_pcs, outcomedf], axis=1)
    X_train_coded_pcs.shape
    cph = CoxPHFitter()
    cph.fit(X_train_coded_pcs, duration_col=time_var, event_col=event_var)
    # Fit the WeibullAFTFitter
    aft = WeibullAFTFitter()
    aft.fit(X_train_coded_pcs, duration_col=time_var, event_col=event_var)
    # add a column names time with values equal to the time column when event is 1 and negative of the time column when event is 0
    X_train_coded_pcs['time']=X_train_coded_pcs[time_var]
    X_train_coded_pcs.loc[X_train_coded_pcs[event_var] == 0,'time' ] = -X_train_coded_pcs[time_var]
    simple_impute_columns_pcs=X_train_coded_pcs.columns.tolist()

    # remove the time and event columns
    simple_impute_columns_pcs.pop(simple_impute_columns_pcs.index(time_var))
    simple_impute_columns_pcs.pop(simple_impute_columns_pcs.index(event_var))
    simple_impute_columns_pcs.pop(simple_impute_columns_pcs.index('time'))
    # remove the time and event columns
    X_train_coded_pcs.drop([time_var, event_var], axis=1, inplace=True)

    imputer_pc = Imputer(num_iter=1, time_limit=30, presets='medium_quality', column_settings={'time': {'time_limit': 500, 'presets': 'best_quality', 'eval_metric': scorefunct_coxPH_custom_scorer}},simple_impute_columns=simple_impute_columns_pcs)
    X_train_coded_pcs.reset_index(drop=True, inplace=True)
    X_train_coded_pcs = TabularDataset(X_train_coded_pcs)
    X_train_coded_pcs.reset_index(drop=True, inplace=True)
    imputer_pc.fit(X_train_coded_pcs)
    
    df_missing_train.dtypes.to_list()
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

    # Impute missing values
    for col in X_test_coded.columns:
        if X_test_coded[col].dtype == "numeric":
            X_test_coded[col] = X_test_coded[col].fillna(X_train_coded[col].mean())
        else:
            X_test_coded[col] = X_test_coded[col].fillna(X_train_coded[col].mode()[0])

    # extract the principal components for the test data
    X_test_coded_pcs=PCA(n_components=10).fit_transform(X_test_coded.drop([time_var, event_var], axis=1))
    # convert the principal components to a data frame
    X_test_coded_pcs = pd.DataFrame(X_test_coded_pcs, columns=[f"PC{i}" for i in range(1, 11)])
    # add the time and event columns to the principal components
    outcomedf=X_test_coded[[time_var, event_var]]
    # drop index
    outcomedf.reset_index(drop=True, inplace=True)
    X_test_coded_pcs.reset_index(drop=True, inplace=True)
    X_test_coded_pcs=pd.concat([X_test_coded_pcs, outcomedf], axis=1)
    
    predtest = cph.predict_expectation(X_test_coded_pcs)

    # Predict using the WeibullAFTFitter
    mean_pred_survival = aft.predict_expectation(X_test_coded_pcs)
    
    # make a copy of the test data
    X_test_coded_pcs_missing=X_test_coded_pcs.copy()
    # put nan in the time column
    X_test_coded_pcs_missing[time_var]=np.nan
    #put 1.0 in the event column
    X_test_coded_pcs_missing[event_var]=1.0
    # drop time and event columns
    X_test_coded_pcs_missing.drop([time_var, event_var], axis=1, inplace=True)
    # add a column names time with nan values
    X_test_coded_pcs_missing['time']=np.nan
    X_test_coded_pcs_missing = TabularDataset(X_test_coded_pcs_missing)
    X_test_coded_pcs_missing = imputer_pc.transform(X_test_coded_pcs_missing)

    # Evaluate
    c_index_imputed = concordance_index(X_test_coded[time_var], -X_test_imputed['time'], X_test_coded[event_var])
    c_index_imputed_pc=concordance_index(X_test_coded[time_var], -X_test_coded_pcs_missing['time'], X_test_coded[event_var])
    c_index_pred = concordance_index(X_test_coded[time_var], predtest, X_test_coded[event_var])
    c_index_weibull = concordance_index(X_test_coded[time_var], mean_pred_survival, X_test_coded[event_var])
    return c_index_imputed,c_index_imputed_pc, c_index_pred, c_index_weibull

##################################################

# Dictionary to store results for each dataset
results = {}

# 10 rounds of experiments
for repi in range(10):
    print(f"Running experiment {repi+1}")
    # Run experiment
    c_index_imputed,c_index_imputed_pc, c_index_pred, c_index_weibull = run_experiment(data)
    # Store results
    results[repi] = {"c_index_imputed": c_index_imputed, "c_index_imputed_pc": c_index_imputed_pc,
                     "c_index_pred": c_index_pred, "c_index_weibull": c_index_weibull}
    # print results
    print(f"Imputed: {c_index_imputed}")
    print(f"Imputed PC: {c_index_imputed_pc}")
    print(f"Predicted: {c_index_pred}")
    print(f"Weibull: {c_index_weibull}")

# print the mean and standard deviation of the results

print("Imputed: ", np.mean([results[i]["c_index_imputed"] for i in results.keys()]), np.std([results[i]["c_index_imputed"] for i in results.keys()]))
print("Imputed PC: ", np.mean([results[i]["c_index_imputed_pc"] for i in results.keys()]), np.std([results[i]["c_index_imputed_pc"] for i in results.keys()]))
print("Predicted: ", np.mean([results[i]["c_index_pred"] for i in results.keys()]), np.std([results[i]["c_index_pred"] for i in results.keys()]))
print("Weibull: ", np.mean([results[i]["c_index_weibull"] for i in results.keys()]), np.std([results[i]["c_index_weibull"] for i in results.keys()]))

