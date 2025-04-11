# AutoFillGluon

AutoFillGluon is a Python package that provides machine learning-based missing data imputation using AutoGluon. It offers a simple and efficient way to handle missing values in your datasets by leveraging the power of AutoGluon's automated machine learning capabilities.

## Features

- Machine learning-based imputation using AutoGluon
- Support for both numeric and categorical data
- Multiple imputation strategies
- Feature importance analysis
- Imputation quality evaluation
- Model saving and loading
- Customizable column settings
- Missingness feature support
- Simple imputation options

## Installation

### Prerequisites

- Python 3.8 or higher
- AutoGluon
- pandas
- numpy
- scikit-learn

### Install

*   **From PyPI (Recommended):**
    ```bash
    # Ensure your conda environment is active
    pip install autofillgluon
    ```

*   **For development (from local clone):**
    ```bash
    # Ensure your conda environment is active before cloning/installing
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
    verbose=True,         # Show progress information
    use_missingness_features=True,  # Add missingness indicators as features
    simple_impute_columns=['category']  # Use simple imputation for categorical columns
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

# Save and load the imputer
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
mean_values = pd.concat(imputed_datasets).groupby(level=0).mean()
std_values = pd.concat(imputed_datasets).groupby(level=0).std()
```

### Evaluating Imputation Quality

```python
# Evaluate imputation quality
results = imputer.evaluate(
    data=complete_df,  # Original complete dataset
    percentage=0.2,    # Percentage of values to set as missing
    n_samples=5        # Number of evaluation samples
)

# Print results
for sample, metrics in results.items():
    print(f"\nSample {sample}:")
    for col, col_metrics in metrics.items():
        print(f"{col}:")
        for metric, value in col_metrics.items():
            print(f"  {metric}: {value:.4f}")
```

### Feature Importance Analysis

```python
# Get feature importance for each imputed column
importance_dict = imputer.feature_importance()

# For a specific column
importance_df = importance_dict['numeric1']
print(importance_df.sort_values(ascending=False).head(10))

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
importance_df.sort_values().tail(10).plot(kind='barh')
plt.title('Feature importance for imputing numeric1')
plt.tight_layout()
plt.show()
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

## Examples

Explore the following examples to learn how to use AutoFillGluon:

1. **Basic Imputation** - `examples/basic/simple_imputation.py`
   - Simple example showing how to impute missing values in a synthetic dataset
   - Demonstrates the core imputation functionality with visualization

2. **Multiple Imputation** - `examples/advanced/multiple_imputation.py`
   - Shows how to generate and analyze multiple imputed datasets
   - Demonstrates handling of imputation uncertainty

3. **Custom Settings** - `examples/advanced/custom_settings.py`
   - Demonstrates column-specific settings and simple imputation
   - Shows how to optimize imputation for different column types

4. **Evaluation** - `examples/evaluation/imputation_evaluation.py`
   - Shows how to evaluate imputation quality
   - Demonstrates various evaluation metrics and visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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
