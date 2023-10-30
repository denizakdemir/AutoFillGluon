## AutoGluonImputer

This package offers a sophisticated solution for handling missing data in datasets using the AutoGluon TabularPredictor. It's adept at working with both numerical and categorical data and provides a machine-learning-driven approach for imputation.

### Prerequisites

Ensure the installation of necessary dependencies:

```sh
pip install --upgrade pandas numpy scikit-learn autogluon
```

Also, don't forget to enable the autoreload extension for seamless notebook experience:

```python
%load_ext autoreload
%reload_ext autoreload
%autoreload 2
```

### Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from autogluon.tabular import TabularDataset
from scripts.autogluonImputer import Imputer
import importlib
```

### Step 2: Prepare the Data

In this example, we'll utilize the Titanic dataset from OpenML:

```python
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True, parser="pandas")

# Combine features and target into one dataframe
df = X.copy()
df['target'] = y

# Drop unnecessary columns
df.drop(['name', 'ticket'], axis=1, inplace=True)

# Ensure correct data types
df = TabularDataset(df)

# Convert object columns to category type
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')

# Convert integer columns to float type
for col in df.columns:
    if df[col].dtype == 'int64':
        df[col] = df[col].astype('float64')

# Split data and introduce missingness
train, test = train_test_split(df, test_size=0.3, random_state=42)
train_missing = train.mask(np.random.random(train.shape) < 0.2)
test_missing = test.mask(np.random.random(test.shape) < 0.2)
```

### Step 3: Impute Missing Values

```python
imputer = Imputer(num_iter=2, time_limit=5)
train_imputed = imputer.fit(train_missing)
test_imputed = imputer.transform(test_missing)
```

### Step 4: Evaluate Imputation

Compare imputed values with the original ones for the `age` column:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
missing_indices_test = test_missing['age'].index[test_missing['age'].apply(np.isnan)]

# Plot imputed values against original values
plt.scatter(test_imputed['age'][missing_indices_test], test['age'][missing_indices_test])
plt.xlabel('Imputed Values')
plt.ylabel('Original Values')
plt.title('Imputed Values vs Original Values')
sns.regplot(x=test_imputed['age'][missing_indices_test], y=test['age'][missing_indices_test], scatter=False, color='red')

# Correlation Coefficient Calculation
df = pd.DataFrame({'imputed': test_imputed['age'][missing_indices_test], 'original': test['age'][missing_indices_test]})
df = df.dropna()
corr = np.corrcoef(df['imputed'], df['original'])[0,1]
plt.text(.6, .75, f'Correlation Coefficient = {round(corr, 2)}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='black')
plt.show()
```

### Multiple Imputation

For cases where multiple imputations are needed:

```python
from scripts.autogluonImputer import multiple_imputation

train_imputed = multiple_imputation(train_missing, n_imputations=10, num_iter=2, time_limit=10, fitonce=True)
```

## Contributing
We welcome contributions! Please see the `Contributing Guide` for guidelines on submitting pull requests, reporting issues, or requesting features.

## License
This project is licensed under the terms of the `Your License`.
