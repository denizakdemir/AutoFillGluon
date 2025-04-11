import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autofillgluon import Imputer

# Create a synthetic dataset with missing values
np.random.seed(42)
n_samples = 1000

# Generate features with different characteristics
x1 = np.random.normal(0, 1, n_samples)  # Complex feature
x2 = 0.5 * x1 + np.random.normal(0, 0.5, n_samples)  # Correlated feature
x3 = np.random.uniform(-1, 1, n_samples)  # Simple uniform feature
x4 = np.random.exponential(1, n_samples)  # Skewed feature

# Create categorical features
categories1 = ['A', 'B', 'C']  # Complex categorical
categories2 = ['X', 'Y']  # Simple binary
cat_feature1 = np.random.choice(categories1, n_samples, p=[0.4, 0.3, 0.3])
cat_feature2 = np.random.choice(categories2, n_samples, p=[0.5, 0.5])

# Create DataFrame
df = pd.DataFrame({
    'complex_feature': x1,
    'correlated_feature': x2,
    'simple_feature': x3,
    'skewed_feature': x4,
    'complex_category': cat_feature1,
    'simple_category': cat_feature2
})

# Introduce missing values with different patterns
missing_patterns = {
    'complex_feature': 0.3,  # More missing values
    'correlated_feature': 0.2,
    'simple_feature': 0.1,  # Fewer missing values
    'skewed_feature': 0.2,
    'complex_category': 0.2,
    'simple_category': 0.1
}

for col, p in missing_patterns.items():
    mask = np.random.random(n_samples) < p
    df.loc[mask, col] = np.nan

print("Original data shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())

# Define custom column settings
column_settings = {
    'complex_feature': {
        'time_limit': 60,  # More time for complex feature
        'eval_metric': 'r2'
    },
    'correlated_feature': {
        'time_limit': 30
    },
    'skewed_feature': {
        'time_limit': 30,
        'eval_metric': 'mae'  # Use MAE for skewed data
    }
}

# Initialize imputer with custom settings
imputer = Imputer(
    num_iter=3,
    time_limit=30,
    verbose=True,
    use_missingness_features=True,
    simple_impute_columns=['simple_feature', 'simple_category'],  # Use simple imputation for these
    column_settings=column_settings
)

# Fit and transform
df_imputed = imputer.fit(df)

print("\nImputed data shape:", df_imputed.shape)
print("\nMissing values after imputation:")
print(df_imputed.isnull().sum())

# Visualize the results
plt.figure(figsize=(15, 10))

# Plot 1: Numeric feature distributions
plt.subplot(2, 3, 1)
sns.kdeplot(data=df['complex_feature'].dropna(), label='Original', alpha=0.5)
sns.kdeplot(data=df_imputed['complex_feature'], label='Imputed', alpha=0.5)
plt.title('Complex Feature Distribution')
plt.legend()

plt.subplot(2, 3, 2)
sns.kdeplot(data=df['correlated_feature'].dropna(), label='Original', alpha=0.5)
sns.kdeplot(data=df_imputed['correlated_feature'], label='Imputed', alpha=0.5)
plt.title('Correlated Feature Distribution')
plt.legend()

plt.subplot(2, 3, 3)
sns.kdeplot(data=df['simple_feature'].dropna(), label='Original', alpha=0.5)
sns.kdeplot(data=df_imputed['simple_feature'], label='Imputed', alpha=0.5)
plt.title('Simple Feature Distribution')
plt.legend()

plt.subplot(2, 3, 4)
sns.kdeplot(data=df['skewed_feature'].dropna(), label='Original', alpha=0.5)
sns.kdeplot(data=df_imputed['skewed_feature'], label='Imputed', alpha=0.5)
plt.title('Skewed Feature Distribution')
plt.legend()

# Plot 2: Categorical distributions
plt.subplot(2, 3, 5)
df['complex_category'].value_counts().plot(kind='bar', alpha=0.5, label='Original')
df_imputed['complex_category'].value_counts().plot(kind='bar', alpha=0.5, label='Imputed')
plt.title('Complex Category Distribution')
plt.legend()

plt.subplot(2, 3, 6)
df['simple_category'].value_counts().plot(kind='bar', alpha=0.5, label='Original')
df_imputed['simple_category'].value_counts().plot(kind='bar', alpha=0.5, label='Imputed')
plt.title('Simple Category Distribution')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate feature importance
importance_dict = imputer.feature_importance(df)

# Print feature importance for each column
print("\nFeature Importance:")
for col, importance_df in importance_dict.items():
    if importance_df is not None:
        print(f"\n{col}:")
        print(importance_df.sort_values('importance', ascending=False).head()) 