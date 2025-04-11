import pytest
import numpy as np
from lifelines.utils import concordance_index as lifelines_cindex # For comparison
from autofillgluon.scorer.scorer import (
    scorefunct_cindex, 
    scorefunct_coxPH, 
    negative_log_likelihood_exponential,
    concordance_index_scorer,
    cox_ph_scorer,
    exponential_nll_scorer
)

# --- Fixtures ---

@pytest.fixture
def survival_data_simple():
    # Negative values indicate censored observations
    # Higher prediction = higher risk (worse outcome)
    y_true = np.array([-1, 2, 1, 4, 5, -2, 7, 8, 9, 10]) # Times: 1, 2, 1, 4, 5, 2, 7, 8, 9, 10; Events: 0, 1, 1, 1, 1, 0, 1, 1, 1, 1
    y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Predictions (risk scores)
    return y_true, y_pred

@pytest.fixture
def survival_data_perfect_concordance():
    # Perfect ranking: higher risk score predicts earlier event time
    y_true = np.array([1, 2, 3, 4, 5]) # All events
    y_pred = np.array([5, 4, 3, 2, 1]) # Higher score = higher risk = earlier event
    return y_true, y_pred

@pytest.fixture
def survival_data_perfect_discordance():
    # Perfect inverse ranking: lower risk score predicts earlier event time
    y_true = np.array([1, 2, 3, 4, 5]) # All events
    y_pred = np.array([1, 2, 3, 4, 5]) # Lower score = lower risk = earlier event (discordant)
    return y_true, y_pred

@pytest.fixture
def survival_data_all_censored():
    y_true = np.array([-1, -2, -3, -4, -5]) # All censored
    y_pred = np.array([1, 2, 3, 4, 5])
    return y_true, y_pred

@pytest.fixture
def survival_data_all_events():
    y_true = np.array([1, 2, 3, 4, 5]) # All events
    y_pred = np.array([5, 4, 3, 2, 1]) # Perfect concordance
    return y_true, y_pred

@pytest.fixture
def survival_data_with_ties():
    # Ties in time and prediction
    y_true = np.array([1, 2, 2, 3, -3]) # Times: 1, 2, 2, 3, 3; Events: 1, 1, 1, 1, 0
    y_pred = np.array([5, 4, 4, 2, 3])  # Predictions (risk scores)
    return y_true, y_pred

# --- Tests for scorefunct_cindex ---

def test_scorefunct_cindex_simple(survival_data_simple):
    y_true, y_pred = survival_data_simple
    cindex = scorefunct_cindex(y_true, y_pred)
    assert 0 <= cindex <= 1
    # Calculate expected using lifelines for comparison (note: lifelines uses lower score = higher risk convention sometimes)
    event_observed = y_true >= 0
    time = np.abs(y_true)
    expected_cindex = lifelines_cindex(time, y_pred, event_observed) 
    np.testing.assert_almost_equal(cindex, expected_cindex)

def test_scorefunct_cindex_perfect(survival_data_perfect_concordance):
    # Note: y_pred=[5, 4, 3, 2, 1] means higher score = higher risk = earlier event
    # Lifelines concordance_index expects higher score = better survival = later event
    # Therefore, perfect inverse ranking according to lifelines convention.
    y_true, y_pred = survival_data_perfect_concordance
    cindex = scorefunct_cindex(y_true, y_pred)
    np.testing.assert_almost_equal(cindex, 0.0) # Expect 0.0 due to lifelines convention

def test_scorefunct_cindex_discordant(survival_data_perfect_discordance):
    # Note: y_pred=[1, 2, 3, 4, 5] means higher score = lower risk = later event
    # This is perfect concordance according to lifelines convention.
    y_true, y_pred = survival_data_perfect_discordance
    cindex = scorefunct_cindex(y_true, y_pred)
    np.testing.assert_almost_equal(cindex, 1.0) # Expect 1.0 due to lifelines convention

def test_scorefunct_cindex_all_censored(survival_data_all_censored):
    y_true, y_pred = survival_data_all_censored
    # C-index is undefined when no events occur, lifelines raises ZeroDivisionError
    with pytest.raises(ZeroDivisionError, match="No admissable pairs"):
        scorefunct_cindex(y_true, y_pred)
    # Alternatively, if we wanted to return NaN or 0.5, the function itself would need modification.
    # For now, testing that it raises the expected error from lifelines is correct.

def test_scorefunct_cindex_all_events(survival_data_all_events):
    # Note: y_pred=[5, 4, 3, 2, 1] means higher score = higher risk = earlier event
    # Perfect inverse ranking according to lifelines convention.
    y_true, y_pred = survival_data_all_events
    cindex = scorefunct_cindex(y_true, y_pred)
    np.testing.assert_almost_equal(cindex, 0.0) # Expect 0.0 due to lifelines convention

def test_scorefunct_cindex_with_ties(survival_data_with_ties):
    y_true, y_pred = survival_data_with_ties
    cindex = scorefunct_cindex(y_true, y_pred)
    assert 0 <= cindex <= 1
    # Compare with lifelines
    event_observed = y_true >= 0
    time = np.abs(y_true)
    expected_cindex = lifelines_cindex(time, y_pred, event_observed)
    np.testing.assert_almost_equal(cindex, expected_cindex)

# --- Tests for scorefunct_coxPH ---

def test_scorefunct_coxPH_simple(survival_data_simple):
    y_true, y_pred = survival_data_simple
    loglik = scorefunct_coxPH(y_true, y_pred)
    assert np.isfinite(loglik)
    # Hard to assert specific value without a known baseline model

def test_scorefunct_coxPH_all_censored(survival_data_all_censored):
    y_true, y_pred = survival_data_all_censored
    loglik = scorefunct_coxPH(y_true, y_pred)
    # If no events, log likelihood should be 0 (negative loglik = 0)
    np.testing.assert_almost_equal(loglik, 0.0)

def test_scorefunct_coxPH_all_events(survival_data_all_events):
    y_true, y_pred = survival_data_all_events
    loglik = scorefunct_coxPH(y_true, y_pred)
    assert np.isfinite(loglik)

def test_scorefunct_coxPH_with_ties(survival_data_with_ties):
    y_true, y_pred = survival_data_with_ties
    loglik = scorefunct_coxPH(y_true, y_pred)
    assert np.isfinite(loglik)

# --- Tests for negative_log_likelihood_exponential ---

def test_nll_exponential_simple(survival_data_simple):
    y_true, y_pred = survival_data_simple
    # Predictions need to be positive rates (lambda)
    y_pred_rates = np.abs(y_pred) + 1e-6 # Ensure positive
    nll = negative_log_likelihood_exponential(y_true, y_pred_rates)
    assert np.isfinite(nll)

def test_nll_exponential_all_censored(survival_data_all_censored):
    y_true, y_pred = survival_data_all_censored
    y_pred_rates = np.abs(y_pred) + 1e-6
    nll = negative_log_likelihood_exponential(y_true, y_pred_rates)
    assert np.isfinite(nll)
    # Check calculation: sum(lambda * time)
    expected_nll = np.sum(y_pred_rates * np.abs(y_true))
    np.testing.assert_almost_equal(nll, expected_nll)

def test_nll_exponential_all_events(survival_data_all_events):
    y_true, y_pred = survival_data_all_events
    y_pred_rates = np.abs(y_pred) + 1e-6
    nll = negative_log_likelihood_exponential(y_true, y_pred_rates)
    assert np.isfinite(nll)
    # Check calculation: sum(lambda*time - log(lambda))
    expected_nll = np.sum(y_pred_rates * np.abs(y_true) - np.log(y_pred_rates))
    np.testing.assert_almost_equal(nll, expected_nll)

def test_nll_exponential_with_ties(survival_data_with_ties):
    y_true, y_pred = survival_data_with_ties
    y_pred_rates = np.abs(y_pred) + 1e-6
    nll = negative_log_likelihood_exponential(y_true, y_pred_rates)
    assert np.isfinite(nll)

# --- Tests for make_scorer objects ---

def test_concordance_index_scorer_properties():
    assert concordance_index_scorer.name == 'concordance_index'
    assert concordance_index_scorer.greater_is_better is True
    assert concordance_index_scorer.optimum == 1

def test_cox_ph_scorer_properties():
    assert cox_ph_scorer.name == 'cox_ph_likelihood'
    # Check the internal sign attribute used by AutoGluon
    assert hasattr(cox_ph_scorer, '_sign'), "Scorer should have a _sign attribute"
    assert cox_ph_scorer._sign == -1, "Sign should be -1 for greater_is_better=False"
    # assert cox_ph_scorer.optimum == 0 # Check optimum if needed

def test_exponential_nll_scorer_properties():
    assert exponential_nll_scorer.name == 'exponential_nll'
    # Check the internal sign attribute used by AutoGluon
    assert hasattr(exponential_nll_scorer, '_sign'), "Scorer should have a _sign attribute"
    assert exponential_nll_scorer._sign == -1, "Sign should be -1 for greater_is_better=False"
    # assert exponential_nll_scorer.optimum == 0 # Check optimum if needed
