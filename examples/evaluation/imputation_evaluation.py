import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autofillgluon import Imputer

# Create a synthetic dataset with known true values
np.random.seed(42)
n_samples = 1000

# Generate features with different characteristics
x1 = np.random.normal(0, 1, n_samples)  # Normal distribution
x2 = 0.5 * x1 + np.random.normal(0, 0.5, n_samples)  # Correlated feature
x3 = np.random.exponential(1, n_samples)  # Skewed feature
x4 = np.random.uniform(-1, 1, n_samples)  # Uniform feature

# Create categorical features
categories1 = ['A', 'B', 'C']  # Multi-category
categories2 = ['X', 'Y']  # Binary
cat_feature1 = np.random.choice(categories1, n_samples, p=[0.4, 0.3, 0.3])
cat_feature2 = np.random.choice(categories2, n_samples, p=[0.5, 0.5])

# Create complete DataFrame (true values)
df_complete = pd.DataFrame({
    'normal_feature': x1,
    'correlated_feature': x2,
    'skewed_feature': x3,
    'uniform_feature': x4,
    'multi_category': cat_feature1,
    'binary_category': cat_feature2
})

# Create missing data patterns
missing_patterns = {
    'normal_feature': 0.2,
    'correlated_feature': 0.2,
    'skewed_feature': 0.2,
    'uniform_feature': 0.2,
    'multi_category': 0.2,
    'binary_category': 0.2
}

# Create dataset with missing values
df_missing = df_complete.copy()
for col, p in missing_patterns.items():
    mask = np.random.random(n_samples) < p
    df_missing.loc[mask, col] = np.nan

print("Complete data shape:", df_complete.shape)
print("\nMissing values per column:")
print(df_missing.isnull().sum())

# Initialize imputer
imputer = Imputer(
    num_iter=3,
    time_limit=30,
    verbose=True,
    use_missingness_features=True
)

# Fit and transform
df_imputed = imputer.fit(df_missing)

# Evaluate imputation quality
evaluation_results = imputer.evaluate(
    data=df_complete,
    percentage=0.2,
    n_samples=5
)

# Print evaluation results
print("\nEvaluation Results:")
for sample, metrics in evaluation_results.items():
    print(f"\nSample {sample}:")
    for col, col_metrics in metrics.items():
        print(f"{col}:")
        for metric, value in col_metrics.items():
            print(f"  {metric}: {value:.4f}")

# Visualize evaluation results
plt.figure(figsize=(15, 10))

# Plot 1: MSE for numeric features
plt.subplot(2, 2, 1)
numeric_mse = []
for sample in evaluation_results:
    for col in ['normal_feature', 'correlated_feature', 'skewed_feature', 'uniform_feature']:
        if 'mse' in evaluation_results[sample][col]:
            numeric_mse.append({
                'Sample': sample,
                'Feature': col,
                'MSE': evaluation_results[sample][col]['mse']
            })
numeric_mse_df = pd.DataFrame(numeric_mse)
sns.boxplot(data=numeric_mse_df, x='Feature', y='MSE')
plt.title('MSE Distribution Across Samples')
plt.xticks(rotation=45)

# Plot 2: MAE for numeric features
plt.subplot(2, 2, 2)
numeric_mae = []
for sample in evaluation_results:
    for col in ['normal_feature', 'correlated_feature', 'skewed_feature', 'uniform_feature']:
        if 'mae' in evaluation_results[sample][col]:
            numeric_mae.append({
                'Sample': sample,
                'Feature': col,
                'MAE': evaluation_results[sample][col]['mae']
            })
numeric_mae_df = pd.DataFrame(numeric_mae)
sns.boxplot(data=numeric_mae_df, x='Feature', y='MAE')
plt.title('MAE Distribution Across Samples')
plt.xticks(rotation=45)

# Plot 3: Accuracy for categorical features
plt.subplot(2, 2, 3)
categorical_acc = []
for sample in evaluation_results:
    for col in ['multi_category', 'binary_category']:
        if 'accuracy' in evaluation_results[sample][col]:
            categorical_acc.append({
                'Sample': sample,
                'Feature': col,
                'Accuracy': evaluation_results[sample][col]['accuracy']
            })
categorical_acc_df = pd.DataFrame(categorical_acc)
sns.boxplot(data=categorical_acc_df, x='Feature', y='Accuracy')
plt.title('Accuracy Distribution Across Samples')
plt.xticks(rotation=45)

# Plot 4: Feature importance
plt.subplot(2, 2, 4)
importance_dict = imputer.feature_importance()
for col in ['normal_feature', 'correlated_feature', 'skewed_feature', 'uniform_feature']:
    importance = importance_dict[col].sort_values(ascending=False).head(5)
    plt.plot(range(len(importance)), importance.values, label=col, marker='o')
plt.title('Top 5 Feature Importance')
plt.xlabel('Rank')
plt.ylabel('Importance Score')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Additional analysis: Compare distributions
plt.figure(figsize=(15, 10))

# Plot distributions for numeric features
for i, col in enumerate(['normal_feature', 'correlated_feature', 'skewed_feature', 'uniform_feature'], 1):
    plt.subplot(2, 2, i)
    sns.kdeplot(data=df_complete[col], label='True', alpha=0.5)
    sns.kdeplot(data=df_imputed[col], label='Imputed', alpha=0.5)
    plt.title(f'{col} Distribution')
    plt.legend()

plt.tight_layout()
plt.show()

# Plot distributions for categorical features
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df_complete['multi_category'].value_counts().plot(kind='bar', alpha=0.5, label='True')
df_imputed['multi_category'].value_counts().plot(kind='bar', alpha=0.5, label='Imputed')
plt.title('Multi-category Distribution')
plt.legend()

plt.subplot(1, 2, 2)
df_complete['binary_category'].value_counts().plot(kind='bar', alpha=0.5, label='True')
df_imputed['binary_category'].value_counts().plot(kind='bar', alpha=0.5, label='Imputed')
plt.title('Binary Category Distribution')
plt.legend()

plt.tight_layout()
plt.show() 