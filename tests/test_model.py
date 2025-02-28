"""Test suite for XGBDistribution model
"""

import os
import pickle

import pytest

import numpy as np
from sklearn.exceptions import NotFittedError

from xgboost_distribution.distributions import AVAILABLE_DISTRIBUTIONS
from xgboost_distribution.model import XGBDistribution


def test_XGBDistribution_early_stopping_fit(small_train_test_data):
    """Integration test to ensure end-to-end functionality"""

    X_train, X_test, y_train, y_test = small_train_test_data
    model = XGBDistribution(distribution="normal", max_depth=3, n_estimators=500)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)

    evals_result = model.evals_result()

    assert model.best_iteration == 6
    assert isinstance(evals_result, dict)


def test_XGBDistribution_early_stopping_predict(small_train_test_data):
    """Check that predict with early stopping uses correct ntrees"""
    X_train, X_test, y_train, y_test = small_train_test_data

    model = XGBDistribution(distribution="normal", max_depth=3, n_estimators=500)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)

    mean, var = model.predict(X_test)
    mean_iter, var_iter = model.predict(
        X_test, iteration_range=(0, model.best_iteration + 1)
    )
    np.testing.assert_array_equal(mean, mean_iter)
    np.testing.assert_array_equal(var, var_iter)


def test_objective_overwrite_by_repeated_fit(small_X_y_data):
    """In the fit step, we set the objective to distribution:normal (e.g.)

    This needs to be re-set at each fit, as it's not a standard xgboost objective.
    """
    X, y = small_X_y_data
    model = XGBDistribution()
    model.fit(X, y)
    model.fit(X, y)


def test_distribution_set_param(small_X_y_data):
    """Check that updating the distribution params works"""
    X, y = small_X_y_data

    model = XGBDistribution(distribution="abnormal")
    with pytest.raises(ValueError):
        model.fit(X, y)  # fails as distribution does not exist

    model.set_params(**{"distribution": "normal"})
    model.fit(X, y)


# -------------------------------------------------------------------------------------
# Failure modes
# -------------------------------------------------------------------------------------


def test_fit_1d_X_fit_fails(small_X_y_data):
    X, y = small_X_y_data
    X = X[..., 0]  # single 1d features

    model = XGBDistribution()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_predict_before_fit_fails(small_X_y_data):
    X, y = small_X_y_data

    model = XGBDistribution()
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_setting_objective_in_init_fails():
    with pytest.raises(ValueError):
        XGBDistribution(objective="binary:logistic")


# -------------------------------------------------------------------------------------
#  Model internal tests
# -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "distribution, expected_margin",
    [
        ("normal", np.array([0.5, np.log(0.5), 0.5, np.log(0.5)])),
        ("poisson", np.array([np.log(0.5), np.log(0.5)])),
    ],
)
def test_get_base_margin(distribution, expected_margin):
    """Test that base_margin are created as expected for each sample"""
    model = XGBDistribution(distribution=distribution)
    X = np.array([[1], [1]])
    y = np.array([1, 0])
    model.fit(X, y)

    margin = model._get_base_margin(n_samples=2)
    np.testing.assert_array_equal(margin, expected_margin)


@pytest.mark.parametrize(
    "distribution",
    list(AVAILABLE_DISTRIBUTIONS.keys()),
)
def test_objective_and_evaluation_funcs_callable(distribution):
    model = XGBDistribution(distribution=distribution)
    assert callable(model._objective_func())
    assert callable(model._evaluation_func())


# -------------------------------------------------------------------------------------
#  Model IO tests
# -------------------------------------------------------------------------------------


def assert_model_equivalence(model_a, model_b, X):
    np.testing.assert_almost_equal(
        model_a._starting_params, model_b._starting_params, decimal=9
    )
    assert model_a._distribution.__class__ == model_b._distribution.__class__

    preds_a = model_a.predict(X)
    preds_b = model_b.predict(X)

    for param_a, param_b in zip(preds_a, preds_b):
        np.testing.assert_array_equal(param_a, param_b)


@pytest.mark.parametrize(
    "model_format",
    ["bst", "json"],
)
def test_XGBDistribution_save_and_load_model(small_X_y_data, model_format, tmpdir):
    X, y = small_X_y_data
    model = XGBDistribution(n_estimators=10)
    model.fit(X, y)

    model_path = os.path.join(tmpdir, f"model.{model_format}")
    model.save_model(model_path)

    saved_model = XGBDistribution()
    saved_model.load_model(model_path)

    assert_model_equivalence(model_a=model, model_b=saved_model, X=X)


def test_XGBDistribution_pickle_dump_and_load(small_X_y_data, tmpdir):
    X, y = small_X_y_data
    model = XGBDistribution(n_estimators=10)
    model.fit(X, y)

    model_path = os.path.join(tmpdir, "model.pkl")

    with open(model_path, "wb") as fd:
        pickle.dump(model, fd)

    with open(model_path, "rb") as fd:
        saved_model = pickle.load(fd)

    assert_model_equivalence(model_a=model, model_b=saved_model, X=X)
