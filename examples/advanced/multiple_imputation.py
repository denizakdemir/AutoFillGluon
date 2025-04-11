import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autofillgluon import multiple_imputation

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

# Generate multiple imputed datasets
imputed_datasets = multiple_imputation(
    data=df,
    n_imputations=5,
    fitonce=True,  # Fit one model and generate multiple imputations
    num_iter=3,
    time_limit=30,
    verbose=True
)

# Calculate statistics across imputations
mean_values = pd.concat(imputed_datasets).groupby(level=0).mean()
std_values = pd.concat(imputed_datasets).groupby(level=0).std()

print("\nMean values across imputations:")
print(mean_values.head())
print("\nStandard deviation across imputations:")
print(std_values.head())

# Visualize the results
plt.figure(figsize=(15, 10))

# Plot 1: Feature distributions across imputations
plt.subplot(2, 2, 1)
for i, dataset in enumerate(imputed_datasets):
    sns.kdeplot(data=dataset['feature1'], label=f'Imputation {i+1}', alpha=0.3)
plt.title('Feature 1 Distribution Across Imputations')
plt.legend()

plt.subplot(2, 2, 2)
for i, dataset in enumerate(imputed_datasets):
    sns.kdeplot(data=dataset['feature2'], label=f'Imputation {i+1}', alpha=0.3)
plt.title('Feature 2 Distribution Across Imputations')
plt.legend()

plt.subplot(2, 2, 3)
for i, dataset in enumerate(imputed_datasets):
    sns.kdeplot(data=dataset['feature3'], label=f'Imputation {i+1}', alpha=0.3)
plt.title('Feature 3 Distribution Across Imputations')
plt.legend()

# Plot 2: Category distribution across imputations
plt.subplot(2, 2, 4)
for i, dataset in enumerate(imputed_datasets):
    dataset['category'].value_counts().plot(kind='bar', alpha=0.3, label=f'Imputation {i+1}')
plt.title('Category Distribution Across Imputations')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate and plot uncertainty
uncertainty = std_values.mean(axis=1)
plt.figure(figsize=(10, 6))
uncertainty.plot(kind='bar')
plt.title('Average Uncertainty Across Imputations')
plt.xlabel('Row Index')
plt.ylabel('Standard Deviation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() 