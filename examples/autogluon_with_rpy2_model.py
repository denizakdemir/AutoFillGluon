import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

from autogluon.tabular import TabularPredictor, TabularDataset

robjects.r('install.packages("randomForest",repos="https://cloud.r-project.org")')


class CustomRModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        print('Entering the `_fit` method')
        # Convert pandas DataFrame to R dataframe
        rdf = robjects.DataFrame({col: robjects.FloatVector(X[col]) for col in X})
        ry = robjects.FloatVector(y)
        robjects.globalenv['train_data'] = rdf
        robjects.globalenv['train_label'] = ry
        # Handle the 'sqrt' value for mtry
        if self.params.get('mtry') == 'sqrt':
            mtry_value = int(np.sqrt(X.shape[1]))
        else:
            mtry_value = self.params.get('mtry')
        
        # Break down the R code execution
        try:
            robjects.r('library(randomForest)')
            robjects.r(f'''
            model <- randomForest(train_data, y=train_label, ntree=500)
            ''')
        except Exception as e:
            print(f"Error encountered: {e}")
            return

        self.model = robjects.globalenv['model']
        print('Exiting the `_fit` method')

    def predict(self, X: pd.DataFrame):
        print('Entering the `predict` method')
        X = self._preprocess(X)
        rdf = robjects.DataFrame({col: robjects.FloatVector(X[:, i]) for i, col in enumerate(X.columns)})
        robjects.globalenv['test_data'] = rdf

        predictions = robjects.r('predict(model, newdata=test_data)')
        return pd.Series(predictions)

    def _set_default_params(self):
        default_params = {
            'ntree': 500
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            get_features_kwargs=dict(
                valid_raw_types=['int', 'float', 'category'],
            )
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params





# Load data
train_data = pd.read_csv('SurvivalDatasets/Lung_data.csv', index_col=0)
train_data.head()
# make names of columns R compatible
train_data.columns = [x.replace('.', '_') for x in train_data.columns]

train_data = TabularDataset(train_data)
label='time'
# Train the custom R model


custom_model = CustomRModel()
y=train_data.loc[:,'time']
# drop time colum, define X
X=train_data.drop('time', axis=1)



# We could also specify hyperparameters to override defaults
custom_model = CustomRModel(hyperparameters={'ntree': 500,
                                             'mtry': 'sqrt',
                                             'replace': True})
custom_model._fit(X=X, y=y)
custom_model.model[0]
custom_model.predict(X=X)

custom_hyperparameters ={CustomRModel:{'ntree': 500}}
#predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters)
predictor = TabularPredictor(label=label).fit(train_data,  presets='best_quality',hyperparameters=custom_hyperparameters)  # We can even use the custom model in a multi-layer stack ensemble


##################
