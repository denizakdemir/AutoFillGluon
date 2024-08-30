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
from autogluon.tabular import TabularDataset
# to evaluate use scikit-learn metrics
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

from autogluon.eda import auto
import contextlib
import sys

import logging
logger = logging.getLogger(__name__)




@contextlib.contextmanager
def redirect_stdout_to_file(file_path):
    """
        Context manager for redirecting standard output to a file.

        This context manager temporarily redirects the standard output (sys.stdout) 
        to the specified file. Any print statements or output generated within 
        the context will be written to the file instead of the console. Once the 
        context is exited, the standard output is restored to its original state, 
        and the file is closed.

        Parameters:
        file_path (str): The path to the file where the standard output will be redirected. 
                        The file is opened in append mode, so existing content will not be overwritten.

        Returns:
        None: This function does not return any value. It only changes the behavior of standard output.

        Usage:
        >>> with redirect_stdout_to_file('output.log'):
        >>>     print('This will be written to output.log instead of the console.')
        >>> print('This will be printed to the console as usual.')
    """
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


import shutil
import pandas as pd
from random import shuffle
import logging
from autogluon.tabular import TabularPredictor

# Setup logging
logger = logging.getLogger(__name__)

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
            logger.error(f"An error occurred while providing the dataset overview: {e}")
            
        
    def fit(self, X_missing):
        logger.info("Fitting the imputer to the data...")
        self.missing_cells = pd.isnull(X_missing)
        X_imputed = X_missing.copy(deep=True)

        if self.use_missingness_features:
            missingness_cols = self.missingness_matrix(X_missing)
            X_imputed = pd.concat([X_imputed, missingness_cols], axis=1)

        # Initial imputation using mean/mode
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
                logger.error(f"Error removing directory: {e.filename} - {e.strerror}.")

            columns = list(X_imputed.columns)
            shuffle(columns)

            for col in columns:
                if (col.endswith('_missing') and self.use_missingness_features) or col in self.simple_impute_columns:
                    continue
                logger.debug(f"Processing column {col}")

                tmp_X = X_imputed.drop(columns=[col])
                mask = self.missing_cells[col] == False

                col_time_limit = self.column_settings.get(col, {}).get('time_limit', self.time_limit)
                col_presets = self.column_settings.get(col, {}).get('presets', self.presets)
                col_label = self.column_settings.get(col, {}).get('label', col)
                col_eval_metric = self.column_settings.get(col, {}).get('eval_metric', None)

                predictor = TabularPredictor(label=col_label, eval_metric=col_eval_metric).fit(
                    tmp_X[mask],
                    time_limit=col_time_limit,
                    presets=col_presets,
                    verbosity=0,
                    num_bag_folds=5,
                    num_bag_sets=1,
                    num_stack_levels=1
                )

                mask_missing = self.missing_cells[col] == True
                if mask_missing.any():
                    X_imputed.loc[mask_missing, col] = predictor.predict(tmp_X[mask_missing])

                self.models[col] = predictor

        # Collect summary statistics or unique categories
        for col in X_imputed.columns:
            if self.col_data_types[col] == 'numeric':
                self.colsummary[col] = {'min': X_imputed[col].min(), 'max': X_imputed[col].max()}
            else:
                self.colsummary[col] = {'categories': X_imputed[col].unique()}
                
        return X_imputed


    def transform(self, X_missing):
        """
            Imputes missing values in the dataset using pre-trained models and initial imputation values.

            This method performs iterative imputation on the dataset, filling in missing values based on models
            trained for each column, and handles categorical variables by ensuring their categories match those
            used during training. It also optionally includes missingness matrix columns if specified.

            Parameters:
            ----------
            X_missing : DataFrame
                The dataset with missing values that need to be imputed. It is assumed to have the same structure
                as the dataset used during model training.

            Returns:
            -------
            DataFrame
                The dataset with imputed values. The imputed values are filled based on predictions from pre-trained
                models and initial imputation values.
        """
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
                        X_imputed.loc[mask_missing, col] = self.models[col].predict(X_imputed.drop(columns=[col]).loc[mask_missing])
                    except Exception as e:
                        logger.error(f"An error occurred while predicting column {col}: {e}")
                else:
                    logger.warning(f"No model found for column {col}. Skipping imputation.")
                        
            # The categorical variables should have the same classes as in the fit method
            for col in columns:
                if self.col_data_types[col] == 'object':
                    X_imputed[col] = X_imputed[col].astype('category')
                    X_imputed[col].cat.set_categories(self.colsummary[col]['categories'])
        return X_imputed


    def feature_importance(self, X_imputed):
        """
            Computes the feature importance for each column using the pre-trained models.

            This method calculates the importance of features in the dataset by evaluating the contribution of
            each feature to the predictive models used for imputation. It returns a dictionary where the keys
            are column names and the values are the corresponding feature importances.

            Parameters:
            ----------
            X_imputed : DataFrame
                The dataset with imputed values. It is used to evaluate feature importance based on the
                pre-trained models.

            Returns:
            -------
            dict
                A dictionary containing feature importances for each column. The keys are the column names, and
                the values are the feature importances as computed by the models. If a model does not support
                feature importance, the corresponding value will be None.
        """
        feature_importances = {}
        for col, model in self.models.items():
            try:
                if hasattr(model, 'feature_importance'):
                    feature_importances[col] = model.feature_importance(X_imputed)
                else:
                    # If the model does not support feature importance
                    feature_importances[col] = None
                    logger.warning(f"Model for column '{col}' does not support feature importance.")
            except Exception as e:
                # Log the error and set feature importance to None
                feature_importances[col] = None
                logger.error(f"An error occurred while computing feature importance for column '{col}': {e}")

        return feature_importances

    def save_models(self, path):
        """
            Saves the trained models and related data to the specified directory.

            This method saves each column's trained model into its own directory, as well as additional
            information used for imputation and model configuration. The saved information includes initial
            imputation values, column summaries, and the list of model columns.

            Parameters:
            ----------
            path : str
                The directory path where the models and related data will be saved. If the directory does not
                exist, it will be created.

            Returns:
            -------
            None
                This method does not return any value. It performs file I/O operations to save models and
                metadata to the specified path.
            
            Notes:
            -----
            - Each column's model is saved in a separate directory named after the column within the specified path.
            - The initial imputation values are saved as a CSV file named 'initial_imputes.csv'.
            - Column summaries are saved using pickle in a file named 'colsummary.pkl'.
            - The list of model columns is saved using pickle in a file named 'model_columns.pkl'.
        """
        # Ensure the base directory exists
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory {path}")

        for col, model in self.models.items():
            # Create a directory for each column's model
            col_path = os.path.join(path, col)
            if not os.path.exists(col_path):
                os.makedirs(col_path)
                logger.info(f"Created directory {col_path}")


            try:
                model.save(str(col_path))
                logger.info(f"Saved model for column '{col}' to {col_path}")
            except Exception as e:
                logger.error(f"Failed to save model for column '{col}': {e}")

        # Save the means and modes for the numeric and categorical columns
        try:
            pd.DataFrame.from_dict(self.initial_imputes, orient='index').to_csv(os.path.join(path, 'initial_imputes.csv'))
            logger.info(f"Saved initial imputes to {path / 'initial_imputes.csv'}")
        except Exception as e:
            logger.error(f"Failed to save initial imputes: {e}")


        # Save the colsummary using pickle
        try:
            with open(os.path.join(path, 'colsummary.pkl'), 'wb') as f:
                pickle.dump(self.colsummary, f)
            logger.info(f"Saved column summary to {path / 'colsummary.pkl'}")
        except Exception as e:
            logger.error(f"Failed to save column summary: {e}")

        # Save model columns
        try:
            with open(os.path.join(path, 'model_columns.pkl'), 'wb') as f:
                pickle.dump(list(self.models.keys()), f)
            logger.info(f"Saved model columns to {path / 'model_columns.pkl'}")
        except Exception as e:
            logger.error(f"Failed to save model columns: {e}")

    def load_models(self, path):
        """
            Loads the trained models and related metadata from the specified directory.

            This method restores the models, initial imputation values, and column summaries from files saved
            in the specified directory. The method also initializes the `self.models` dictionary with the
            column names and loads the models for each column from their respective directories.

            Parameters:
            ----------
            path : str
                The directory path from which the models and related data will be loaded.

            Returns:
            -------
            None
                This method does not return any value. It performs file I/O operations to load models and
                metadata from the specified path and update the instance attributes.

            Notes:
            -----
            - The column names are loaded from a pickle file named 'model_columns.pkl'.
            - The models are loaded from directories named after each column within the specified path.
            - The initial imputation values are loaded from a CSV file named 'initial_imputes.csv'.
            - The column summaries are loaded using pickle from a file named 'colsummary.pkl'.
        """
        try:
            # Load column names first
            logger.info("Loading model column names from 'model_columns.pkl'")
            with open(os.path.join(path, 'model_columns.pkl'), 'rb') as f:
                model_columns = pickle.load(f)
            
            logger.info("Successfully loaded model column names")
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
        except Exception as e:
            logger.error(f"Failed to load models, initial imputes, and column summary: {e}")

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
                logger.warning(f"No model found for column {col}. Skipping...")
                continue


            # non_missing_indices = modified_data[col].dropna().index.tolist() # old code
            non_missing_indices = modified_data[col].notna().index # new code replace it with above line it will fast the code because we are using numpy rather than normal list

            n_missing = int(len(non_missing_indices) * percentage)
            logger.info(f"Adding missingness to column {col}: {n_missing} values will be set to NaN")

            missing_indices = np.random.choice(non_missing_indices, size=n_missing, replace=False)
            # Use .loc to modify the DataFrame directly
            modified_data.loc[missing_indices, col] = np.nan
            missingness_indices[col] = missing_indices.tolist()
            logger.info(f"Missingness added to column {col}")

        logger.info(f"Missingness added to column {col}")
        return modified_data, missingness_indices


    def evaluate_imputation(self, data, percentage, ntimes=10):
        """
            Evaluates the imputation performance of the model by measuring the accuracy or error metrics on imputed data.

            This method introduces missing values into the dataset at random, imputes the missing values using
            the trained models, and evaluates the imputation performance by comparing the imputed values to the
            true values. The evaluation is repeated multiple times to get a more reliable measure of performance.

            Parameters:
            ----------
            data : DataFrame
                The original dataset containing the true values. This dataset is used to evaluate the performance of
                the imputation by comparing the imputed values against the true values.
            percentage : float
                The percentage of data to randomly introduce as missing in each iteration. Should be between 0 and 100.
            ntimes : int, optional, default=10
                The number of times to repeat the evaluation process. More iterations can provide a more robust estimate
                of the imputation performance.

            Returns:
            -------
            dict
                A dictionary where each key is the iteration number and each value is another dictionary containing
                evaluation metrics for each column. For categorical columns, the metrics include 'accuracy'. For numerical
                columns, the metrics include 'mse' (mean squared error) and 'mae' (mean absolute error).

            Notes:
            -----
            - The `add_missingness_at_random` method is used to introduce missing values into the dataset.
            - The `transform` method is used to impute the missing values.
            - The evaluation metrics are computed based on the type of data in each column: 'object' (categorical) or numeric.
            - For categorical columns, accuracy is used as the evaluation metric.
            - For numeric columns, mean squared error (MSE) and mean absolute error (MAE) are computed.
            - Rows with NaN values in the predictions or true values are excluded from the evaluation.  
        """
        accuracies = {}

        for rep in range(ntimes):
            results = {}
            modified_data, missingness_indices = self.add_missingness_at_random(data, percentage)
            logger.info(f"Missing data introduced in {percentage}% of the dataset")
            imputed_data = self.transform(modified_data)
            logger.info("Data imputation completed")

            for col in data.columns:
                if col not in self.models:
                    logger.warning(f"No model found for column {col}. Skipping...")
                    continue
                y_true_subset = data.loc[missingness_indices[col], col]
                y_pred_subset = imputed_data.loc[missingness_indices[col], col]

                # Filter out NaN values from predictions
                valid_indices = ~y_true_subset.isna()
                y_true_subset = y_true_subset[valid_indices]
                y_pred_subset = y_pred_subset[valid_indices]
                assert isinstance(y_true_subset, (pd.Series))
                assert isinstance(y_pred_subset, (pd.Series))
                assert y_true_subset.shape == y_pred_subset.shape
                # combine y_true_subset and y_pred_subset into one dataframe
                combdf=pd.concat([y_true_subset, y_pred_subset], axis=1)
                # add column names
                combdf.columns=['y_true', 'y_pred']
                combdf=TabularDataset(combdf)
                # drop rows with NaN values

                logger.info(f"Evaluating column {col}")

                combdf=combdf.dropna()
                if self.col_data_types[col] == 'object':
                    evaluation_acc_score = accuracy_score(combdf['y_true'], combdf['y_pred'])
                    evaluation = {'accuracy': evaluation_acc_score}
                    logger.info(f"Accuracy for column {col}: {evaluation_acc_score:.4f}")
                else:
                    evaluation_mse_score = mean_squared_error(combdf['y_true'], combdf['y_pred'])
                    evaluation_mae_score = mean_absolute_error(combdf['y_true'], combdf['y_pred'])
                    evaluation = {'mse': evaluation_mse_score, 'mae': evaluation_mae_score}
                    logger.info(f"MSE for column {col}: {evaluation_mse_score:.4f}, MAE for column {col}: {evaluation_mae_score:.4f}")

                results[col] = evaluation
            accuracies[rep] = results
        logger.info("Imputation evaluation completed")
        return accuracies




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
        logger.info("Fitting the model once and imputing multiple times...")
        imputer = Imputer(**kwargs)
        imputer.fit(data)
        for i in range(n_imputations):
            try:
                imputed_data = imputer.transform(data)  # Ensure transform method is supported
                imputed_datasets.append(imputed_data)
            except Exception as e:
                logger.error(f"Error during imputation {i+1}: {e}")
                break
    else:
        for i in range(n_imputations):
            logger.debug(f"Performing imputation {i+1}/{n_imputations}")
            try:
                imputer = Imputer(**kwargs)
                imputed_data = imputer.fit(data)  # Fit and transform TODO: Change the name of this method to fit_transform if it able to handle fit and transform operations
                imputed_datasets.append(imputed_data)
            except Exception as e:
                logger.error(f"Error during imputation {i+1}: {e}")
                break
    return imputed_datasets

