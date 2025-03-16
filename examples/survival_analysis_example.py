"""
Example demonstrating the use of AutoFillGluon for survival analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines.datasets import load_rossi
from lifelines import KaplanMeierFitter
from autogluon.tabular import TabularPredictor, TabularDataset

# Import scorers from autofillgluon
from autofillgluon import (
    cox_ph_scorer, 
    concordance_index_scorer, 
    exponential_nll_scorer
)
from autofillgluon import Imputer


def prepare_survival_data(df, time_col, event_col):
    """
    Prepare survival data for AutoGluon by encoding time and event.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing survival data
    time_col : str
        Column name for the time variable
    event_col : str
        Column name for the event indicator (1 = event, 0 = censored)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with a 'time' column encoding both time and event
        (positive = event, negative = censored)
    """
    # Create a copy to avoid modifying the original
    df_model = df.copy()
    
    # Create the time column (positive for events, negative for censored)
    df_model['time'] = df_model[time_col]
    df_model.loc[df_model[event_col] == 0, 'time'] = -df_model.loc[df_model[event_col] == 0, time_col]
    
    # Drop the original time and event columns
    df_model = df_model.drop(columns=[time_col, event_col])
    
    return df_model


def plot_survival_curves(df, time_col, event_col, group_col=None):
    """Plot Kaplan-Meier survival curves."""
    kmf = KaplanMeierFitter()
    
    plt.figure(figsize=(10, 6))
    
    if group_col is None:
        # Plot one curve for the whole dataset
        kmf.fit(df[time_col], event_observed=df[event_col], label="All")
        kmf.plot()
    else:
        # Plot a curve for each group
        for group, group_df in df.groupby(group_col):
            kmf.fit(group_df[time_col], event_observed=group_df[event_col], label=f"{group_col}={group}")
            kmf.plot()
    
    plt.title("Kaplan-Meier Survival Curves")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    

def evaluate_predictions(y_true, y_true_event, y_pred):
    """
    Evaluate survival predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True survival times
    y_true_event : array-like
        Event indicators (1 = event, 0 = censored)
    y_pred : array-like
        Predicted risk scores
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    from lifelines.utils import concordance_index
    
    # For concordance_index, higher predictions should indicate higher risk
    # (shorter survival times), so we use the negative of predictions
    c_index = concordance_index(y_true, -y_pred, event_observed=y_true_event)
    
    return {
        'concordance_index': c_index
    }


def main():
    print("Loading and preparing the Rossi recidivism dataset...")
    # Load the Rossi recidivism dataset
    rossi = load_rossi()
    
    # Convert week to float (in case it's integer)
    rossi['week'] = rossi['week'].astype(float)
    
    # Display basic info about the dataset
    print(f"\nDataset shape: {rossi.shape}")
    print("\nColumn descriptions:")
    print("- week: Week of first arrest after release or end of study")
    print("- arrest: Arrested during study period? (1=yes, 0=no)")
    print("- fin: Financial aid received? (1=yes, 0=no)")
    print("- age: Age at release (years)")
    print("- race: Race (1=black, 0=other)")
    print("- wexp: Work experience (1=yes, 0=no)")
    print("- mar: Married? (1=yes, 0=no)")
    print("- paro: Released on parole? (1=yes, 0=no)")
    print("- prio: Number of prior convictions")
    
    # Look at the first few rows
    print("\nFirst 5 rows of the dataset:")
    print(rossi.head())
    
    # Plot the Kaplan-Meier survival curves
    print("\nPlotting Kaplan-Meier survival curves...")
    plt.figure(figsize=(12, 8))
    
    # Overall survival curve
    plt.subplot(2, 2, 1)
    plot_survival_curves(rossi, 'week', 'arrest')
    plt.title("Overall Survival (Time to Arrest)")
    
    # Stratified by financial aid
    plt.subplot(2, 2, 2)
    plot_survival_curves(rossi, 'week', 'arrest', 'fin')
    plt.title("Survival by Financial Aid")
    
    # Stratified by work experience
    plt.subplot(2, 2, 3)
    plot_survival_curves(rossi, 'week', 'arrest', 'wexp')
    plt.title("Survival by Work Experience")
    
    # Stratified by marital status
    plt.subplot(2, 2, 4)
    plot_survival_curves(rossi, 'week', 'arrest', 'mar')
    plt.title("Survival by Marital Status")
    
    plt.tight_layout()
    plt.savefig('survival_curves.png')
    print("Saved survival curves to 'survival_curves.png'")
    
    # Prepare the dataset for AutoGluon
    print("\nPreparing data for AutoGluon...")
    df_model = prepare_survival_data(rossi, 'week', 'arrest')
    
    # Create an artificial version with some missing values
    print("\nCreating version with missing values for demonstration...")
    # Create a mask with 15% missing values
    np.random.seed(42)
    mask = np.random.random(df_model.shape) < 0.15
    
    # Create a copy with missing values
    df_missing = df_model.copy()
    for i in range(df_missing.shape[0]):
        for j in range(df_missing.shape[1]):
            # Don't add missing values to the target column
            if j != df_missing.columns.get_loc('time') and mask[i, j]:
                df_missing.iloc[i, j] = np.nan
    
    # Impute missing values using AutoFillGluon
    print("\nImputing missing values...")
    imputer = Imputer(num_iter=2, time_limit=15, verbose=True)
    df_imputed = imputer.fit(df_missing)
    
    print(f"\nOriginal shape: {df_model.shape}")
    print(f"Missing data shape: {df_missing.shape}")
    print(f"Imputed data shape: {df_imputed.shape}")
    
    # Train models using different survival metrics
    print("\nTraining survival models with different metrics...")
    
    # First, convert to TabularDataset for AutoGluon
    df_model = TabularDataset(df_model)
    df_imputed = TabularDataset(df_imputed)
    
    # Define common training parameters
    common_params = {
        'label': 'time',
        'time_limit': 60,
        'presets': 'medium_quality',
        'verbosity': 0
    }
    
    # Train with Cox PH scorer
    print("\nTraining with Cox PH scorer...")
    cox_predictor = TabularPredictor(eval_metric=cox_ph_scorer, **common_params)
    cox_predictor.fit(df_model)
    
    # Train with concordance index scorer
    print("\nTraining with Concordance Index scorer...")
    cindex_predictor = TabularPredictor(eval_metric=concordance_index_scorer, **common_params)
    cindex_predictor.fit(df_model)
    
    # Train with exponential NLL scorer
    print("\nTraining with Exponential NLL scorer...")
    exp_predictor = TabularPredictor(eval_metric=exponential_nll_scorer, **common_params)
    exp_predictor.fit(df_model)
    
    # Make predictions
    print("\nMaking predictions with all models...")
    cox_preds = cox_predictor.predict(df_model)
    cindex_preds = cindex_predictor.predict(df_model)
    exp_preds = exp_predictor.predict(df_model)
    
    # Evaluate models
    print("\nEvaluating models...")
    cox_eval = evaluate_predictions(rossi['week'], rossi['arrest'], cox_preds)
    cindex_eval = evaluate_predictions(rossi['week'], rossi['arrest'], cindex_preds)
    exp_eval = evaluate_predictions(rossi['week'], rossi['arrest'], exp_preds)
    
    print(f"Cox PH scorer C-index: {cox_eval['concordance_index']:.4f}")
    print(f"Concordance Index scorer C-index: {cindex_eval['concordance_index']:.4f}")
    print(f"Exponential NLL scorer C-index: {exp_eval['concordance_index']:.4f}")
    
    # Train a model on the imputed data
    print("\nTraining model on imputed data...")
    imputed_predictor = TabularPredictor(eval_metric=cox_ph_scorer, **common_params)
    imputed_predictor.fit(df_imputed)
    
    # Compare the leaderboards
    print("\nComparing model leaderboards:")
    print("\nOriginal data leaderboard:")
    print(cox_predictor.leaderboard(df_model, silent=True)[['model', 'score', 'pred_time_val']])
    
    print("\nImputed data leaderboard:")
    print(imputed_predictor.leaderboard(df_imputed, silent=True)[['model', 'score', 'pred_time_val']])
    
    # Plot risk scores vs. actual times
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(rossi['week'], -cox_preds, c=rossi['arrest'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Event (1=arrest)')
    plt.xlabel('Time (weeks)')
    plt.ylabel('Risk Score (Cox PH)')
    plt.title('Cox PH Risk Scores')
    
    plt.subplot(1, 3, 2)
    plt.scatter(rossi['week'], -cindex_preds, c=rossi['arrest'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Event (1=arrest)')
    plt.xlabel('Time (weeks)')
    plt.ylabel('Risk Score (C-index)')
    plt.title('C-index Risk Scores')
    
    plt.subplot(1, 3, 3)
    plt.scatter(rossi['week'], -exp_preds, c=rossi['arrest'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Event (1=arrest)')
    plt.xlabel('Time (weeks)')
    plt.ylabel('Risk Score (Exp NLL)')
    plt.title('Exponential NLL Risk Scores')
    
    plt.tight_layout()
    plt.savefig('risk_scores.png')
    print("Saved risk score plots to 'risk_scores.png'")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()