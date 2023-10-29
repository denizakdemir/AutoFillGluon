import numpy as np
import pandas as pd
from  lifelines.metrics import concordance_index
import lifelines.metrics as metrics
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularDataset


def scorefunct_cindex(y_true, y_pred):
    """
    Function to compute the concordance index between two arrays
    :param y_true: array of true values (negative if censored and positive if not)
    :param y_pred: array of predicted values
    :return: concordance index
    """
    
    event=y_true>=0
    event=event.astype(int)
    time=np.abs(y_true)

    return concordance_index(time, y_pred, event_observed=event)

# concordance_index_custom_scorer = make_scorer(name='concordance_index',
#                                                   score_func=scorefunct_cindex,
#                                                   optimum=1,
#                                                   greater_is_better=True)


def scorefunct_coxPH(y_true, y_pred):
    # Determine which observations are events
    event = y_true >= 0
    event = event.astype(int)
    
    # Extract the time (absolute value)
    time = np.abs(y_true)
    # Sort data by time
    sort_idx = np.argsort(time)
    # convert to list
    sort_idx = sort_idx.tolist()

    y_pred = y_pred[sort_idx]
    event = event[sort_idx]
    time = time[sort_idx]

    risk_set = []
    log_lik = 0
    
    for i in range(len(time)):
        if event[i] == 1:
            # Compute the risk set at the current event time
            risk_set = np.where(time >= time[i])[0]
            
            # Compute the log likelihood for the current event
            log_lik += y_pred[i] - np.log(np.sum(np.exp(y_pred[risk_set])))
    
    return -log_lik  # We return the negative log likelihood

# coxPH_custom_scorer = make_scorer(name='coxPH', score_func=scorefunct_coxPH,
#                                                   optimum=-np.inf,
#                                                   greater_is_better=False)



def negative_log_likelihood_exponential(y_true, y_pred):
    """
    Function to compute the negative log-likelihood for exponential data with right-censored observations.
    :param y_true: array of true values (negative if censored and positive if not)
    :param y_pred: array of predicted values (µ values)
    :return: negative log-likelihood
    """

    event = y_true >= 0
    event = event.astype(int)
    y_observed = np.abs(y_true)

    # Calculate the negative log-likelihood
    neg_log_likelihood = -np.sum(event * (np.log(y_pred) - y_observed / y_pred))

    return neg_log_likelihood





# example that dont run when loading the module
if __name__ == '__main__':
    # example
    y_true = np.array([-1, 2, 1, 4, 5, 1, 7, 8, 9, 10])
    y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(scorefunct_cindex(y_true, y_pred))
    #print(concordance_index_custom_scorer(y_true, y_pred))


    # Example usage
    y_true = np.array([2, -3, 5, -6, 4])
    y_pred = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    print(scorefunct_coxPH(y_true, y_pred))
    #print(coxPH_custom_scorer(y_true, y_pred))

    # Example usage with TabularPredictor
    from autogluon.tabular import TabularPredictor
    from lifelines.datasets import load_rossi

    df = load_rossi()
    df['week'] = df['week'].astype(float)

    df_model=df.copy()

    # add a column time that is the the week column with negative values if censored and positive if not
    df_model['time'] = df_model['week']
    df_model.loc[df_model['arrest'] == 0, 'time'] = -df_model['week']

    # drop week and arrest columns
    df_model=df_model.drop(columns=['week', 'arrest'])

    # train model
    label = 'time'
    coxPH_custom_scorer=make_scorer(name='coxPH', score_func=scorefunct_coxPH, optimum=-np.inf, greater_is_better=False)
    predictor = TabularPredictor(label=label, eval_metric=coxPH_custom_scorer).fit(df_model, presets="good_quality")
    predictor.leaderboard(df_model, silent=True)
    predictions=predictor.predict(df_model)

    # compute concordance index
    print(concordance_index(df['week'], -predictions, event_observed=df['arrest']))

    # plot df['week'] against -predictions
    from matplotlib import pyplot as plt
    plt.scatter(df['week'], -predictions, c=df['arrest'])
    plt.xlabel('week')
    plt.ylabel('predictions')
    
    plt.show()

    # another example
    from lifelines.datasets import load_leukemia
    df = load_leukemia()
    df['t'] = df['t'].astype(float)

    df_model=df.copy()

    # add a column time that is the the week column with negative values if censored and positive if not
    df_model['time'] = df_model['t']
    df_model.loc[df_model['status'] == 0, 'time'] = -df_model['t']

    # drop t and status columns
    df_model=df_model.drop(columns=['t', 'status'])

    # train model
    label = 'time'
    df_model=TabularDataset(df_model)
    predictor = TabularPredictor(label=label, eval_metric=coxPH_custom_scorer).fit(df_model, presets="best_quality", time_limit=1000)
    predictor.leaderboard(df_model, silent=True)
    predictions=predictor.predict(df_model)

    # compute concordance index
    print(concordance_index(df['t'], -predictions, event_observed=df['status']))

    ##################
    negative_log_likelihood_exponential_custom_scorer = make_scorer(name='negative_log_likelihood_exponential',
                                                        score_func=negative_log_likelihood_exponential, 
                                                        optimum=-np.inf,
                                                        greater_is_better=False)

    predictor = TabularPredictor(label=label, eval_metric=negative_log_likelihood_exponential_custom_scorer).fit(df_model, presets="best_quality", time_limit=1000)
    predictor.leaderboard(df_model, silent=True)
    predictions=predictor.predict(df_model)

    # compute concordance index
    print(concordance_index(df['t'], -predictions, event_observed=df['status']))
   

    -predictions



