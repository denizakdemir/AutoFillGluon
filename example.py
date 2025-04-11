import pandas as pd
import numpy as np
from autofillgluon.imputer.imputer import Imputer

# Create a sample dataset with missing values
def create_sample_data():
    np.random.seed(42)
    n_samples = 100
    
    # Create numeric columns with missing values
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'salary': np.random.normal(50000, 20000, n_samples),
        'experience': np.random.normal(10, 5, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    mask = np.random.random(df.shape) < 0.2  # 20% missing values
    df[mask] = np.nan
    
    return df

def main():
    # Create sample data
    print("Creating sample dataset...")
    df = create_sample_data()
    print("\nOriginal dataset with missing values:")
    print(df.head())
    print("\nMissing value counts:")
    print(df.isnull().sum())
    
    # Initialize imputer
    print("\nInitializing imputer...")
    imputer = Imputer(
        num_iter=3,
        time_limit=30,
        use_missingness_features=True,
        simple_impute_columns=['education']  # Use simple imputation for education
    )
    
    # Fit the imputer
    print("\nFitting imputer...")
    imputed_df = imputer.fit(df)
    
    # Show results
    print("\nImputed dataset:")
    print(imputed_df.head())
    print("\nMissing values after imputation:")
    print(imputed_df.isnull().sum())
    
    # Save and load models
    print("\nSaving models...")
    imputer.save_models("saved_models")
    
    print("\nLoading models...")
    new_imputer = Imputer()
    new_imputer.load_models("saved_models")
    
    # Test transform with new data
    print("\nTesting transform with new data...")
    new_data = create_sample_data()  # Create new data with missing values
    transformed_df = new_imputer.transform(new_data)
    
    print("\nTransformed dataset:")
    print(transformed_df.head())
    print("\nMissing values after transform:")
    print(transformed_df.isnull().sum())
    
    # Evaluate imputation
    print("\nEvaluating imputation...")
    evaluation_results = imputer.evaluate_imputation(df, percentage=0.2, ntimes=2)
    print("\nEvaluation results:")
    for rep, results in evaluation_results.items():
        print(f"\nRepetition {rep + 1}:")
        for col, metrics in results.items():
            print(f"{col}: {metrics}")

if __name__ == "__main__":
    main() 