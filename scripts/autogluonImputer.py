# Standard library imports
import os
import shutil
import pickle
from random import shuffle

# Third-party imports
import pandas as pd
import numpy as np

# AutoGluon imports
from autogluon.tabular import TabularPredictor
from autogluon.eda import auto
import contextlib
import sys


@contextlib.contextmanager
def redirect_stdout_to_file(file_path):
    # Backup the reference to the original standard output
    original_stdout = sys.stdout

    # Open the file in append mode and set it as the new standard output
    file = open(file_path, 'a')
    sys.stdout = file

    try:
        yield
    finally:
        # Restore the original standard output and close the file
        sys.stdout = original_stdout
        file.close()



class Imputer:
    def __init__(self, num_iter=10, time_limit=60,  presets=['medium_quality', 'optimize_for_deployment'], column_settings=None, use_missingness_features=False, simple_impute_columns=[]):
        """
        Imputer leveraging AutoGluon for predictive imputation of missing data.

        Uses separate AutoGluon models for each column with missing values to predict and impute based on other columns.

        Parameters
        ----------
        num_iter : int, optional (default=10)
            Number of iterations for the imputation process.

        time_limit : int, optional (default=60)
            Time in seconds for each individual model training during the imputation.

        presets : list[str] or str, optional (default=['medium_quality', 'optimize_for_deployment'])
            Configuration presets to use with AutoGluon. For detailed options, refer to the AutoGluon documentation.

        column_settings : dict, optional (default=None)
            Fine-tuning for specific columns. Allows setting time limits, presets per column and evaluation metric, e.g., 
            {'column_name': {'time_limit': 120, 'presets': 'best_quality', 'eval_metric': 'roc_auc'}}.

        use_missingness_features : bool, optional (default=False)
            If True, additional features indicating the missingness of other columns will be added to the dataset during imputation.

        Attributes
        ----------
        col_data_types : dict
            Dictionary mapping column names to their data types ('numeric' or 'object').

        initial_imputes : dict
            Initial imputation values for each column, either mean (for numeric columns) or mode (for object columns).

        models : dict
            Dictionary storing trained AutoGluon models for each column with missing data.

        missing_cells : pandas.DataFrame
            DataFrame indicating the location of missing values in the original input.

        colsummary : dict
            Summary statistics or unique values for each column, assisting in consistent imputation during the transform step.

        Methods
        -------
        dataset_overview(train_data, test_data, label)
            Provides an overview of the dataset.

        fit(X_missing)
            Fits the imputer to the data with missing values, training individual models for each column.

        transform(X_missing)
            Predictively imputes missing values on a new dataset using the models trained during the fit method.

        feature_importance(X_imputed)
            Provides feature importance for each trained model.

        save_models(path)
            Saves the trained models and other relevant attributes to the specified path.

        load_models(path)
            Loads the models and other relevant attributes from the specified path.

        missingness_matrix(data)
            If use_missingness_features=True, this method will generate a matrix indicating the missingness of the columns.
        """

        self.num_iter = num_iter
        self.time_limit = time_limit
        self.presets = presets
        self.col_data_types = {}
        self.initial_imputes = {}
        self.models = {}
        self.colsummary = {}
        self.column_settings = column_settings or {}
        self.use_missingness_features = use_missingness_features
        self.simple_impute_columns = simple_impute_columns 
        
    def dataset_overview(self, train_data, test_data, label):
        """
        Provide an overview of the dataset.
        
        Parameters:
        ----------
        train_data : DataFrame
            The training dataset.
        test_data : DataFrame
            The testing dataset.
        label : str
            The target variable.
            
        Returns:
        -------
        None. Prints the overview of the dataset.
        """
        try:
            auto.dataset_overview(train_data=train_data, test_data=test_data, label=label)
        except Exception as e:
            print(f"An error occurred while providing the dataset overview: {e}")
        
    def fit(self, X_missing):
        '''
        Fit the imputer to the data.

        Parameters
        ----------
        X_missing : pandas DataFrame

        eval_metrics : dict of str -> str or None. the keys are the column names and the values are the evaluation metrics to use for each column.
            The default is None, which means that AutoGluon will choose the evaluation metric for each column.  
            The evaluation metrics are the same as those used in AutoGluon for classification and regression.
            See https://auto.gluon.ai/stable/index.html for more details.

        Returns
        -------
        X_imputed : pandas DataFrame
        '''
        print("Fitting the imputer to the data...")
        with redirect_stdout_to_file('autogluon_fit.log'):
            self.missing_cells = pd.isnull(X_missing)
            X_imputed = X_missing.copy(deep=True)  # Create a deep copy
            # Add missingness matrix columns if use_missingness_features is True
            if self.use_missingness_features:
                missingness_cols = self.missingness_matrix(X_missing)
                X_imputed = pd.concat([X_imputed, missingness_cols], axis=1)

            for col in X_imputed.columns:
                if X_imputed[col].dtype == 'object' or str(X_imputed[col].dtype) == 'category':
                    self.col_data_types[col] = 'object'
                    mode_value = X_imputed[col].mode()[0]
                    X_imputed[col].fillna(mode_value, inplace=True)
                    self.initial_imputes[col] = mode_value
                else:
                    self.col_data_types[col] = 'numeric'
                    mean_value = X_imputed[col].mean()
                    X_imputed[col].fillna(mean_value, inplace=True)
                    self.initial_imputes[col] = mean_value

            for iter in range(self.num_iter):
                try:
                    shutil.rmtree('AutogluonModels')
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                print(f"Iteration {iter + 1}")
                columns = list(X_imputed.columns)
                shuffle(columns)

                for col in columns:
                    # Skip if the column is from the missingness matrix or is in simple_impute_columns
                    if (col.endswith('_missing') and self.use_missingness_features) or col in self.simple_impute_columns:
                        continue
                    print(f"Processing column {col}")

                    tmp_X = X_imputed.copy()
                    mask = self.missing_cells[col] == False

                    col_time_limit = self.column_settings.get(col, {}).get('time_limit', self.time_limit)
                    col_presets = self.column_settings.get(col, {}).get('presets', self.presets)
                    col_label = self.column_settings.get(col, {}).get('label', col)  # Default to the current column if not specified
                    col_eval_metric = self.column_settings.get(col, {}).get('eval_metric', None)  # Default to None if not specified

                    predictor = TabularPredictor(label=col_label, eval_metric=col_eval_metric).fit(
                        tmp_X[mask],
                        time_limit=col_time_limit,
                        presets=col_presets,
                        verbosity=0,
                        num_bag_folds=5,
                        num_bag_sets=3,
                        num_stack_levels=1
                    )

                    mask_missing = self.missing_cells[col] == True
                    if X_missing[col].isnull().any():
                        X_imputed.loc[mask_missing, col] = predictor.predict(tmp_X[mask_missing])

                    self.models[col] = predictor
            # for each numerical variable in the dataset get the mins and maxes, for each categorical variable get the possible categories.
            # this is needed for the transform method
            self.colsummary = {}
            for col in X_imputed.columns:
                if self.col_data_types[col] == 'numeric':
                    self.colsummary[col] = {'min': X_imputed[col].min(), 'max': X_imputed[col].max()}
                else:
                    self.colsummary[col] = {'categories': X_imputed[col].unique()}
        return X_imputed

    def transform(self, X_missing):
        X_imputed = X_missing.copy(deep=True)
        # Add missingness matrix columns if use_missingness_features is True
        if self.use_missingness_features:
            missingness_cols = self.missingness_matrix(X_missing)
            X_imputed = pd.concat([X_imputed, missingness_cols], axis=1)

        # Fill the initial values
        for col in X_imputed.columns:
            X_imputed[col].fillna(self.initial_imputes[col], inplace=True)

        for iter in range(self.num_iter):
            columns = list(X_imputed.columns)
            shuffle(columns)

            for col in columns:
                # Skip if the column is from the missingness matrix or is in simple_impute_columns
                if (col.endswith('_missing') and self.use_missingness_features) or col in self.simple_impute_columns:
                    continue
                mask_missing = pd.isnull(X_missing[col])

                # Skip the prediction if there are no missing values in the current column
                if not mask_missing.any():
                    continue

                if col in self.models:
                    try:
                        X_imputed.loc[mask_missing, col] = self.models[col].predict(X_imputed[mask_missing])
                    except Exception as e:
                        print(f"An error occurred while predicting column {col}: {e}")
                else:
                    print(f"No model found for column {col}. Skipping imputation.")
                        
            # The categorical variables should have the same classes as in the fit method
            for col in columns:
                if self.col_data_types[col] == 'object':
                    X_imputed[col] = X_imputed[col].astype('category')
                    X_imputed[col].cat.set_categories(self.colsummary[col]['categories'], inplace=True)
        return X_imputed

    def feature_importance(self, X_imputed):
        feature_importances = {}
        for col in self.models:
            feature_importances[col] = self.models[col].feature_importance(X_imputed)
        return feature_importances

    def save_models(self, path):
        # Ensure the base directory exists
        if not os.path.exists(path):
            os.makedirs(path)

        for col, model in self.models.items():
            # Create a directory for each column's model
            col_path = os.path.join(path, col)
            if not os.path.exists(col_path):
                os.makedirs(col_path)
            model.save(col_path)

        # Save the means and modes for the numeric and categorical columns
        pd.DataFrame.from_dict(self.initial_imputes, orient='index').to_csv(os.path.join(path, 'initial_imputes.csv'))

        # Save the colsummary using pickle
        with open(os.path.join(path, 'colsummary.pkl'), 'wb') as f:
            pickle.dump(self.colsummary, f)

        # Save model columns
        with open(os.path.join(path, 'model_columns.pkl'), 'wb') as f:
            pickle.dump(list(self.models.keys()), f)

    def load_models(self, path):
        # Load column names first
        with open(os.path.join(path, 'model_columns.pkl'), 'rb') as f:
            model_columns = pickle.load(f)

        # Initialize the self.models dictionary with placeholders (None)
        self.models = {col: None for col in model_columns}

        # Load the actual models from their respective directories
        for col in self.models:
            self.models[col] = TabularPredictor.load(os.path.join(path, col))

        # Load the means and modes for the numeric and categorical columns
        self.initial_imputes = pd.read_csv(os.path.join(path, 'initial_imputes.csv'), index_col=0, header=None).to_dict()[1]

        # Load the colsummary
        with open(os.path.join(path, 'colsummary.pkl'), 'rb') as f:
            self.colsummary = pickle.load(f)

    def add_missingness_at_random(self, data, percentage):
        """
        Add missingness at random at a specified percentage without overlapping with original missingness.

        Parameters
        ----------
        data : pandas.DataFrame
            The input data.
        percentage : float
            The percentage of values to set to NaN.

        Returns
        -------
        pandas.DataFrame
            The data with added missingness.
        dict
            Dictionary containing indices of additional missingness for each column.
        """
        modified_data = data.copy(deep=True)
        missingness_indices = {}

        for col in modified_data.columns:
            if col not in self.models:
                continue
            non_missing_indices = modified_data[col].dropna().index.tolist()
            n_missing = int(len(non_missing_indices) * percentage)

            missing_indices = np.random.choice(non_missing_indices, n_missing, replace=False)
            # Use .loc to modify the DataFrame directly
            modified_data.loc[missing_indices, col] = np.nan
            missingness_indices[col] = missing_indices.tolist()

        return modified_data, missingness_indices


    def evaluate_imputation(self, data, percentage, ntimes=10):
        """
        Evaluate the imputation process by introducing missingness and comparing the accuracy.

        Parameters
        ----------
        data : pandas.DataFrame
            The input data.
        percentage : float
            The percentage of values to set to NaN.
        ntimes : int, optional (default=10)

        Returns
        -------
        dict
            Dictionary of average accuracies for each column.
        """
        accuracies = {}

        for rep in range(ntimes):

            results = {}
            modified_data, missingness_indices = self.add_missingness_at_random(data, percentage)
            imputed_data = self.transform(modified_data)

            for col in data.columns:
                if col not in self.models:
                    continue
                y_true_subset = data.loc[missingness_indices[col]]
                y_pred_subset = imputed_data.loc[missingness_indices[col]]
                evaluation = self.models[col].evaluate_predictions(y_true=y_true_subset, y_pred=y_pred_subset, auxiliary_metrics=True)
                results[col] = evaluation

            accuracies[rep] = results

        meanaccuracies = {}
        for col in data.columns:
            if col not in self.models:
                continue
            meanaccuracies[col] = np.mean([accuracies[rep][col] for rep in accuracies])

        return meanaccuracies


def multiple_imputation(data, n_imputations=5, fitonce=False, **kwargs):
    """
    Perform multiple imputation on a dataset using the Imputer class.
    
    Parameters:
    ----------
    data : pandas DataFrame
        The dataset with missing values to be imputed.
    n_imputations : int, optional (default=5)
        Number of multiple imputations to be performed.
    fitonce : bool, optional (default=False)
        If True, fit the model only once and use it for all imputations.
    **kwargs : 
        Arguments to be passed to the Imputer class.
    
    Returns:
    -------
    imputed_datasets : list of pandas DataFrame
        List of datasets with imputed values.
    """

    imputed_datasets = []

    if fitonce:
        print("Fitting the model once and imputing multiple times...")
        imputer = Imputer(**kwargs)
        imputer.fit(data)
        for i in range(n_imputations):
            print(f"Performing imputation {i+1}/{n_imputations}")
            imputed_data = imputer.transform(data)  # Assuming Imputer has a transform method
            imputed_datasets.append(imputed_data)
    else:
        for i in range(n_imputations):
            print(f"Performing imputation {i+1}/{n_imputations}")
            imputer = Imputer(**kwargs)
            imputed_data = imputer.fit(data)  # Fit and transform
            imputed_datasets.append(imputed_data)
    return imputed_datasets


