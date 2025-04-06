# AutoFillGluon

Machine learning-based missing data imputation using AutoGluon.

## Overview

AutoFillGluon provides a sophisticated approach to missing data imputation using machine learning models from [AutoGluon](https://auto.gluon.ai/). Instead of simple mean/mode imputation or traditional methods, AutoFillGluon trains a separate prediction model for each column with missing values, leveraging information from other columns to make intelligent predictions.

### Features

- **Machine learning-based imputation** using AutoGluon's predictive models
- **Iterative refinement** for improved imputation quality
- **Handles both numerical and categorical data** automatically
- **Multiple imputation support** to account for imputation uncertainty
- **Built-in evaluation** of imputation quality 
- **Survival analysis integration** with custom scoring functions

## Installation

You can install AutoFillGluon from PyPI:

```bash
pip install autofillgluon
```

For development, install directly from GitHub:

```bash
pip install git+https://github.com/denizakdemir/AutoFillGluon.git
```

Or clone and install locally:

```bash
git clone https://github.com/denizakdemir/AutoFillGluon.git
cd AutoFillGluon
pip install -e .
```

## Quick Start

### Basic Imputation

```python
import pandas as pd 
import numpy as np
from autofillgluon import Imputer

# Load your dataset with missing values
# df = pd.read_csv('your_data.csv')

# For this example, create data with missing values
df = pd.DataFrame({
    'numeric1': [1.0, 2.0, np.nan, 4.0, 5.0],
    'numeric2': [10.0, np.nan, 30.0, 40.0, 50.0],
    'category': ['A', 'B', np.nan, 'A', 'B']
})

# Initialize the imputer with desired settings
imputer = Imputer(
    num_iter=3,           # Number of iterations for imputation refinement
    time_limit=30,        # Time limit per column model (seconds)
    verbose=True          # Show progress information
)

# Fit the imputer and get imputed data
df_imputed = imputer.fit(df)

# For new data, use transform method
new_df = pd.DataFrame({
    'numeric1': [6.0, np.nan, 8.0],
    'numeric2': [np.nan, 70.0, 80.0],
    'category': ['C', 'A', np.nan]
})

new_df_imputed = imputer.transform(new_df)

# You can save and reload the imputer for later use
imputer.save('my_imputer')
loaded_imputer = Imputer.load('my_imputer')
```

### Multiple Imputation

Generate multiple imputed datasets to account for imputation uncertainty:

```python
from autofillgluon import multiple_imputation

# Create 5 imputed versions of the dataset
imputed_datasets = multiple_imputation(
    data=df,
    n_imputations=5,
    fitonce=True,     # Fit one model and generate multiple imputations
    num_iter=3,
    time_limit=30
)

# Calculate statistics across imputations
combined_results = pd.concat([
    dataset['numeric1'].apply(lambda x: x**2) 
    for dataset in imputed_datasets
])
mean_results = combined_results.groupby(level=0).mean()
std_results = combined_results.groupby(level=0).std()
```

### Survival Analysis Support

AutoFillGluon includes custom scoring functions for survival analysis with AutoGluon:

```python
from autofillgluon import cox_ph_scorer
from autogluon.tabular import TabularPredictor
from lifelines.datasets import load_rossi

# Load Rossi recidivism dataset
df = load_rossi()

# Prepare data for survival analysis
df_model = df.copy()
df_model['time'] = df_model['week']
df_model.loc[df_model['arrest'] == 0, 'time'] = -df_model['week']
df_model = df_model.drop(columns=['week', 'arrest']) 

# Train model with Cox PH loss
predictor = TabularPredictor(
    label='time',
    eval_metric=cox_ph_scorer
).fit(
    df_model,
    presets="high_quality",
    time_limit=120
)

# Make predictions
risk_scores = predictor.predict(df_model)
```

## Evaluation

Evaluate imputation quality on complete data by artificially introducing missingness:

```python
import matplotlib.pyplot as plt
from autofillgluon import plot_imputation_evaluation

# Assume we have complete data without missing values
complete_df = pd.DataFrame(...)

# Train imputer
imputer = Imputer(num_iter=3, time_limit=30)
imputer.fit(complete_df)

# Evaluate imputation performance
metrics = imputer.evaluate(
    data=complete_df,
    percentage=0.2,   # Percentage of values to set as missing
    n_samples=5       # Number of evaluation samples
)

# Print metrics
for col, results in metrics.items():
    print(f"Column: {col}")
    for metric, values in results.items():
        print(f"  {metric}: mean={values['mean']:.4f}, std={values['std']:.4f}")

# Visualize imputation for a specific column
# First, create data with artificial missingness
missing_data, missing_indices = imputer._create_missing_data(complete_df, 0.2)
imputed_data = imputer.transform(missing_data)

# Create mask indicating which values were missing
mask = pd.DataFrame(
    False, 
    index=complete_df.index, 
    columns=complete_df.columns
)
for col, indices in missing_indices.items():
    mask.loc[indices, col] = True

# Plot evaluation for a numeric column
plot_imputation_evaluation(
    original_df=complete_df, 
    imputed_df=imputed_data, 
    missing_mask=mask,
    column='numeric1'
)
```

## Advanced Features

### Custom Column Settings

Specify different settings for each column:

```python
imputer = Imputer(
    num_iter=3,
    time_limit=30,
    column_settings={
        'important_col': {
            'time_limit': 120,            # More time for this column
            'presets': 'best_quality',    # Better quality setting
            'eval_metric': 'r2'           # Specific evaluation metric
        },
        'simple_col': {
            'time_limit': 10              # Less time for this column
        }
    },
    # Skip ML imputation for some columns (use mean/mode only)
    simple_impute_columns=['simple_col_2', 'simple_col_3']
)
```

### Feature Importance Analysis

Analyze which features are most important for imputing each column:

```python
# After fitting
importance_by_column = imputer.feature_importance()

# For a specific column
importance_df = importance_by_column['numeric1']
importance_df.sort_values(ascending=False).head(10)

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
importance_df.sort_values().tail(10).plot(kind='barh')
plt.title('Feature importance for imputing numeric1')
plt.tight_layout()
plt.show()
```

## Examples

Explore the following examples to learn how to use AutoFillGluon:

1. **Basic Imputation** - `examples/basic/simple_imputation.py`
   - Simple example showing how to impute missing values in a synthetic dataset
   - Demonstrates the core imputation functionality with visualization

2. **California Housing Dataset** - `examples/california_housing_example.py`
   - Real-world example using the California Housing dataset
   - Shows how to evaluate imputation quality on data with artificially introduced missingness

3. **Survival Analysis** - `examples/survival_analysis_example.py`
   - Advanced example integrating imputation with survival analysis
   - Demonstrates using custom scorers for survival prediction tasks

To run an example:
```bash
# Navigate to the repository root
cd AutoFillGluon

# Run a specific example
python examples/basic/simple_imputation.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use AutoFillGluon in your research, please cite:

```
@software{autofillgluon,
  author = {Akdemir, Deniz},
  title = {AutoFillGluon: Machine Learning based Missing Data Imputation},
  url = {https://github.com/denizakdemir/AutoFillGluon},
  year = {2023},
}
```

## Acknowledgements

- [AutoGluon](https://auto.gluon.ai/) - The underlying machine learning framework
- [lifelines](https://lifelines.readthedocs.io/) - Survival analysis functionality