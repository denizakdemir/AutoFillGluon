# AutoGluon-Based Imputer Package

## Overview
This package offers a sophisticated solution for handling missing data in datasets using the AutoGluon TabularPredictor. It's adept at working with both numerical and categorical data and provides a machine-learning-driven approach for imputation.

Key features include predictive imputation, customizable iteration settings, and multiple imputation capabilities, among others. The package is also capable of evaluating imputation quality and analyzing feature importance post-imputation.

## Installation
Ensure the installation of necessary dependencies:

```sh
pip install --upgrade pandas numpy scikit-learn autogluon
```

## Usage

### Example with Real Data (California Housing Dataset)

In this example, we'll use the California Housing dataset, introducing random missingness and then applying the imputer to fill in these gaps.

#### Step 1: Import Libraries and Load Data
```python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from autogluonImputer import Imputer 
from autogluon.tabular import TabularDataset

```

#### Step 2: Prepare the Data
```python
# Load the data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target
# convert HouseAge to categorical
# first convert to integer. But we have missing so we use a map
# map
HouseAge=df['HouseAge'].map(lambda x: int(x))
HouseAge
df['HouseAge']=HouseAge
df['HouseAge']
# now convert to categorical
df['HouseAge']=df['HouseAge'].astype('category')
df.dtypes
# Split the data into train and test sets
train, test = train_test_split(df, test_size=0.3, random_state=42)

# Introduce missingness
train_missing = train.mask(np.random.random(train.shape) < 0.2)
test_missing = test.mask(np.random.random(test.shape) < 0.2)
```

#### Step 3: Impute Missing Values
```python
imputer = Imputer()
train_imputed = imputer.fit(train_missing)
test_imputed = imputer.transform(test_missing)
```

#### Step 4: Evaluate Imputation
```python
# Compare imputed values with original values for the target variable
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Identify missing indices in test dataset
missing_indices_test = test_missing['target'].index[test_missing['target'].apply(np.isnan)]

# Plot imputed values against original values
plt.scatter(test_imputed['target'][missing_indices_test], test['target'][missing_indices_test])
plt.xlabel('Imputed Values')
plt.ylabel('Original Values')
plt.title('Imputed Values vs Original Values')
sns.regplot(x=test_imputed['target'][missing_indices_test], y=test['target'][missing_indices_test], scatter=False, color='red')
# Calculate and display the correlation coefficient
corr = np.corrcoef(test_imputed['target'][missing_indices_test], test['target'][missing_indices_test])[0,1]
plt.text(.6, .75, f'Correlation Coefficient = {round(corr, 2)}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='black')
plt.show()
```

## Advanced Usage

For more complex scenarios, such as applying column-specific settings, using missingness features, or performing multiple imputations, refer to the [Advanced Usage section](#advanced-usage) in the main documentation.

## Contributing
We welcome contributions! Please see the `Contributing Guide` for guidelines on submitting pull requests, reporting issues, or requesting features.

## License
This project is licensed under the terms of the `Your License`.