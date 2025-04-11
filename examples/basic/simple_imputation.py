import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autofillgluon import Imputer

# Create a synthetic dataset with missing values
np.random.seed(42)
n_samples = 1000

# Generate correlated features
x1 = np.random.normal(0, 1, n_samples)
x2 = 0.5 * x1 + np.random.normal(0, 0.5, n_samples)
x3 = -0.3 * x1 + 0.7 * x2 + np.random.normal(0, 0.3, n_samples)

# Create a categorical feature
categories = ['A', 'B', 'C']
cat_feature = np.random.choice(categories, n_samples, p=[0.4, 0.3, 0.3])

# Create DataFrame
df = pd.DataFrame({
    'feature1': x1,
    'feature2': x2,
    'feature3': x3,
    'category': cat_feature
})

# Introduce missing values
missing_mask = np.random.random(df.shape) < 0.2
df[missing_mask] = np.nan

print("Original data shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())

# Initialize and fit the imputer
imputer = Imputer(
    num_iter=3,
    time_limit=30,
    verbose=True,
    use_missingness_features=True
)

# Fit and transform
df_imputed = imputer.fit(df)

print("\nImputed data shape:", df_imputed.shape)
print("\nMissing values after imputation:")
print(df_imputed.isnull().sum())

# Visualize the results
plt.figure(figsize=(15, 10))

# Plot 1: Feature distributions before and after imputation
plt.subplot(2, 2, 1)
sns.kdeplot(data=df['feature1'].dropna(), label='Original', alpha=0.5)
sns.kdeplot(data=df_imputed['feature1'], label='Imputed', alpha=0.5)
plt.title('Feature 1 Distribution')
plt.legend()

plt.subplot(2, 2, 2)
sns.kdeplot(data=df['feature2'].dropna(), label='Original', alpha=0.5)
sns.kdeplot(data=df_imputed['feature2'], label='Imputed', alpha=0.5)
plt.title('Feature 2 Distribution')
plt.legend()

plt.subplot(2, 2, 3)
sns.kdeplot(data=df['feature3'].dropna(), label='Original', alpha=0.5)
sns.kdeplot(data=df_imputed['feature3'], label='Imputed', alpha=0.5)
plt.title('Feature 3 Distribution')
plt.legend()

# Plot 2: Category distribution
plt.subplot(2, 2, 4)
df['category'].value_counts().plot(kind='bar', alpha=0.5, label='Original')
df_imputed['category'].value_counts().plot(kind='bar', alpha=0.5, label='Imputed')
plt.title('Category Distribution')
plt.legend()

plt.tight_layout()
plt.show()

# Get feature importance
try:
    importance_dict = imputer.feature_importance(df)
    for col, importance_df in importance_dict.items():
        if importance_df is not None:
            print(f"\nFeature importance for {col}:")
            # Sort by importance score and get top 5
            top_features = importance_df.sort_values(by='importance', ascending=False).head()
            print(top_features)
except ValueError as e:
    print(f"\nCould not get feature importance: {str(e)}")
except Exception as e:
    print(f"\nUnexpected error getting feature importance: {str(e)}") 