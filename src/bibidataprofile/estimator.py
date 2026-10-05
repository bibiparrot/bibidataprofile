# -*- coding: utf-8 -*-
"""most used estimators

"""
import joblib
import pandas as pd
from optbinning import Scorecard, BinningProcess
from optbinning.scorecard import plot_auc_roc, plot_ks
from pyexpat import features

from patsy.test_state import test_Center
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, get_scorer, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
# from xgboost import XGBRegressor, XGBClassifier
# from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# from bibifactor.base import (
#     Estimator,
#     ks_scorer,
#     ks_score,
#     ranking_precision_score_1_basis_points,
#     plot_precision_recall_curve,
#     plot_roc_curve,
#     kendalltau_score,
#     spearmanr_score,
#     EstimatorType,
# )
from loguru import logger
import numpy as np

from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
from optbinning.scorecard.plots import _check_arrays, _check_parameters
from scipy.stats import ks_2samp, kendalltau, spearmanr
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_recall_curve,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import _check_y

from bibidataprofile.dataprofile import read_data_file


def load_and_predict(model_path, test_X):
    estimator = joblib.load(pathlib.Path(model_path) / "model_all.pkl")
    model_all_test_y = estimator.predict(test_X)
    estimator = joblib.load(pathlib.Path(model_path) / "model_train.pkl")
    model_test_y = estimator.predict(test_X)
    return model_all_test_y, model_test_y


def estimate_and_save(train_X, train_y, test_X, test_y, estimator_type, model_path):
    """estimate a model
    Args:
        train_X (pd.DataFrame): training table in dataframe.
        train_y (pd.Series): training target in series.
        test_X (pd.DataFrame): testing table in dataframe.
        test_y (pd.Series): testing target in series.
        estimator_type (str): estimator type, 'classification' or 'regression'.
    Returns:
        Estimator: estimator instance.
    """
    if estimator_type == EstimatorType.Regression:
        estimator = Regression()
    elif estimator_type == EstimatorType.Classification:
        estimator = Classification()

    if not pathlib.Path(model_path).exists():
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

    (train_score, test_score) = estimator.fit_with_evaluation(train_X, train_y, test_X, test_y)
    joblib.dump(estimator, pathlib.Path(model_path) / "model_train.pkl")
    X = pd.concat([train_X, test_X])
    y = pd.concat([train_y, test_y])
    estimator.fit(X, y)
    joblib.dump(estimator, pathlib.Path(model_path) / "model_all.pkl")
    return train_score, test_score


class Base:
    def __sklearn_is_fitted__(self) -> bool:
        # return hasattr(self, "_Booster")
        raise NotFittedError(
            f"This {self.__class__.__name__} instance should be fitted. Call"
            f"'fit()' with proper arguments."
        )

    @staticmethod
    def check_X_y(X, y):
        if not isinstance(X, pd.DataFrame):
            X, y = check_X_y(X, y)
            X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas.DataFrame or numpy.ndarray.")

        if not isinstance(y, pd.Series):
            y = _check_y(y)
            y = pd.Series(y, name="y")

        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("X must be a pandas.Series or numpy.ndarray.")
        return X.copy(deep=True), y

    @staticmethod
    def check_array(X):
        if not isinstance(X, pd.DataFrame):
            X = check_array(X)
            X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas.DataFrame or numpy.ndarray.")
        return X.copy(deep=True)


class BaseFactorAnalyzer(Base, BaseEstimator, metaclass=ABCMeta):
    """base class for factor analyzer in scikit-learn fit style.

    Note:
        can use 'fit'.

    """

    @abstractmethod
    def fit(self, X, y, *args, **kwargs) -> "BaseFactorAnalyzer":
        """Fit the factor analysis according to the given training data."""


@dataclass(frozen=True)
class EstimatorType:
    """types to tag classification and regression.

    EstimatorType.Classification
    EstimatorType.Regression

    Note:
        can use 'classification' or 'regression' instead.

    """

    Classification = "classification"
    Regression = "regression"


class Estimator(Base, BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y, *args, **kwargs) -> "Estimator":
        """Fit the estimator according to the given training data (X, y)"""

    @abstractmethod
    def fit_with_params(self, X, y, params, *args, **kwargs) -> "Estimator":
        """Fit the estimator according to the given training data (X, y) and params"""

    @abstractmethod
    def fit_with_evaluation(
            self, X_train, y_train, X_test, y_test, *args, **kwargs
    ) -> float:
        """fit / train with (X_train, y_train), and evaluate with (X_test, y_test)

        Args:
            X_train : pandas.DataFrame
                training table in dataframe.
            y_train: pandas.Series
                training target in series.
            X_test : pandas.DataFrame
                testing table in dataframe.
            y_test: pandas.Series
                testing target in series.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            float: the evaluation score in [0, 1], the larger the better.
        """
        pass

    @abstractmethod
    def predict(self, X, *args, **kwargs) -> pd.Series:
        """Fit the estimator according to the given training data (X, y)"""

    @abstractmethod
    def predict_proba(self, X, *args, **kwargs) -> pd.Series:
        """Fit the estimator according to the given training data (X, y) and params"""

    def best_params_greedy(
            self,
            X_train,
            y_train,
            X_test=None,
            y_test=None,
            cv_n_splits: int = 3,
            params_grid: dict = None,
            initial_params: dict = None,
            *args,
            **kwargs,
    ) -> (dict, float, float):
        """find the best parameters from params grid using greedy methods.

        Args:
            X_train : pandas.DataFrame
                training table in dataframe.
            y_train: pandas.Series
                training target in series.
            X_test : pandas.DataFrame
                testing table in dataframe.
            y_test: pandas.Series
                testing target in series.
            cv_n_splits: int
                cross validation n(cv_n_splits) split.
            params_grid: dict
                candidate parameters grid.
            initial_params: dict
                initial parameters.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple(dict, float, float): best_params, train_score, test_score
        """
        pass

    @staticmethod
    def optimal_threshold(y_true, y_prob):
        scores = []
        thresholds = np.arange(0, 1, 0.005)
        for threshold in thresholds:
            score = f1_score(y_true, y_prob > threshold)
            scores.append(score)
        best_threshold = thresholds[np.argmax(scores)]
        return np.max(scores), best_threshold

    @staticmethod
    def f1_smart(y_true, y_pred):
        args = np.argsort(y_pred)
        tp = y_true.sum()
        fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(
            y_true.shape[0] + tp - 1, tp, -1
        )
        res_idx = np.argmax(fs)
        return 2 * fs[res_idx], (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2


def ks_optbinning(y_true, y_pred):
    """Calculating the Kolmogorov-Smirnov score using optbinning event view"""
    y_true, y_pred = _check_arrays(y_true, y_pred)

    n_samples = y_true.shape[0]
    n_event = np.sum(y_true)
    n_nonevent = n_samples - n_event

    idx = y_pred.argsort()
    yy = y_true[idx]
    # pp = y_pred[idx]

    cum_event = np.cumsum(yy)
    cum_population = np.arange(0, n_samples)
    cum_nonevent = cum_population - cum_event

    p_event = cum_event / n_event
    p_nonevent = cum_nonevent / n_nonevent

    p_diff = p_nonevent - p_event
    ks_score = np.max(p_diff)
    return ks_score


def ks_scipy(y_true, y_pred):
    """Calculating the Kolmogorov-Smirnov score using scipy ks_2samp"""
    ks_score = ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
    return ks_score


def ks_score(y_true, y_pred):
    """Calculating the Kolmogorov-Smirnov score using scikit learn roc_curve"""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    return max(tpr - fpr)


ks_scorer = make_scorer(ks_score, needs_proba=True)


def kendalltau_score(y_true, y_pred):
    """Calculating the ranking consistency score by kendalltau"""
    score, pvalue = kendalltau(y_true, y_pred)
    return score


def spearmanr_score(y_true, y_pred):
    """Calculating the ranking consistency score by spearmanr"""
    score, pvalue = spearmanr(y_true, y_pred)
    return score


def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int / float
        Rank.

    Returns
    -------
    precision @k : float
    """
    y_true = _check_y(y_true)
    y_score = _check_y(y_score)
    if isinstance(k, float) and k < 1:
        k = int(len(y_true) * k)

    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)


def ranking_precision_score_1_basis_points(y_true, y_score):
    return ranking_precision_score(y_true, y_score, 0.01)


def plot_precision_recall_curve(
        y, y_pred, title=None, xlabel=None, ylabel=None, savefig=False, fname=None, **kwargs
):
    """Plot precision recall curve.
    see: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py

    Parameters
    ----------
    y : array-like, shape = (n_samples,)
        Array with the target labels.

    y_pred : array-like, shape = (n_samples,)
        Array with predicted probabilities.

    title : str or None, optional (default=None)
        Title for the plot.

    xlabel : str or None, optional (default=None)
        Label for the x-axis.

    ylabel : str or None, optional (default=None)
        Label for the y-axis.

    savefig : bool (default=False)
        Whether to save the figure.

    fname : str or None, optional (default=None)
        Name for the figure file.

    **kwargs : keyword arguments
        Keyword arguments for matplotlib.pyplot.savefig().
    """
    y, y_pred = _check_arrays(y, y_pred)

    _check_parameters(title, xlabel, ylabel, savefig, fname)

    # Define the arrays for plotting
    precision, recall, _ = precision_recall_curve(y, y_pred)

    # Define the plot settings
    if title is None:
        title = "Precision Recall curve"
    if xlabel is None:
        xlabel = "Recall"
    if ylabel is None:
        ylabel = "Precision"

    PrecisionRecallDisplay(precision=precision, recall=recall).plot()

    plt.title(title, fontdict={"fontsize": 14})
    plt.xlabel(xlabel, fontdict={"fontsize": 12})
    plt.ylabel(ylabel, fontdict={"fontsize": 12})
    plt.legend(loc="lower right")

    # Save figure if requested. Pass kwargs.
    if savefig:
        plt.savefig(fname=fname, **kwargs)
        plt.close()


def plot_roc_curve(
        y, y_pred, title=None, xlabel=None, ylabel=None, savefig=False, fname=None, **kwargs
):
    """Plot Roc Auc curve.
        see
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

        see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay

        ----------
        y : array-like, shape = (n_samples,)
            Array with the target labels.

        y_pred : array-like, shape = (n_samples,)
            Array with predicted probabilities.

        title : str or None, optional (default=None)
            Title for the plot.

        xlabel : str or None, optional (default=None)
            Label for the x-axis.

        ylabel : str or None, optional (default=None)
            Label for the y-axis.

        savefig : bool (default=False)
            Whether to save the figure.

        fname : str or None, optional (default=None)
            Name for the figure file.

        **kwargs : keyword arguments
            Keyword arguments for matplotlib.pyplot.savefig().
    """
    y, y_pred = _check_arrays(y, y_pred)

    _check_parameters(title, xlabel, ylabel, savefig, fname)

    # Define the plot settings
    if title is None:
        title = "Roc curve"
    if xlabel is None:
        xlabel = "False Positive Rate"
    if ylabel is None:
        ylabel = "True Positive Rate"

    RocCurveDisplay.from_predictions(y, y_pred).plot()

    plt.title(title, fontdict={"fontsize": 14})
    plt.xlabel(xlabel, fontdict={"fontsize": 12})
    plt.ylabel(ylabel, fontdict={"fontsize": 12})
    plt.legend(loc="lower right")

    # Save figure if requested. Pass kwargs.
    if savefig:
        plt.savefig(fname=fname, **kwargs)
        plt.close()


class Regression(Estimator):
    """
    class for wrapping regression estimator


    ...

    Attributes
    ----------
    model_class : class
        e.g.  sklearn.ensemble.RandomForestRegressor, xgboost.XGBRegressor, catboost.CatBoostRegressor
    params : dict
        params to initialize estimator_class
    scoring : str
        scoring method see https://scikit-learn.org/stable/modules/model_evaluation.html

    Methods
    -------
    fit(X, y):
        train by X, y
    """

    def __init__(
            self, model_class=None, params: dict = None, scoring: str = None, verbose=False
    ):
        if model_class is None:
            model_class = LGBMRegressor
        self.model_class = model_class
        self.params = params if params is not None else {}
        self.scoring = scoring if scoring is not None else "r2"
        self.scorer = get_scorer(self.scoring)
        self.verbose = verbose
        self.model = None
        self.feature_names = None
        if self.verbose:
            logger.info(f"estimator = {self.model_class}, scorer = {self.scorer}")
        self.estimator_type = EstimatorType.Regression

    def __sklearn_is_fitted__(self) -> bool:
        return self.model is not None and self.model.__sklearn_is_fitted__()

    def fit(self, X, y, *args, **kwargs) -> "Estimator":
        """Fit the estimator according to the given training data (X, y)"""
        self.fit_with_params(X, y, params=self.params)
        return self

    def fit_with_params(self, X, y, params, *args, **kwargs) -> "Estimator":
        """Fit the estimator according to the given training data (X, y)"""
        X, y = self.check_X_y(X, y)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X.columns = [f'X{i}' for i in range(len(X.columns))]
        self.model = self.model_class(**params)
        self.model.fit(X, y)
        if self.verbose:
            score = self.scorer(self.model, X, y)
            logger.info(f"training score = {score}, scorer = {self.scorer}")
        return self

    def fit_with_evaluation(
            self, X_train, y_train, X_test, y_test, *args, **kwargs
    ) -> (float, float):
        """fit / train with (X_train, y_train), and evaluate with (X_test, y_test)

        Args:
            X_train : pandas.DataFrame
                training table in dataframe.
            y_train: pandas.Series
                training target in series.
            X_test : pandas.DataFrame
                testing table in dataframe.
            y_test: pandas.Series
                testing target in series.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            float: the evaluation score in [0, 1], the larger the better.
            float: the evaluation score in [0, 1], the larger the better.
        """
        self.fit(X_train, y_train)
        train_score = self.scorer(self.model, X_train, y_train)
        test_score = self.scorer(self.model, X_test, y_test)
        if self.verbose:
            logger.info(f"testing score = {test_score}, scorer = {self.scorer}")
        return train_score, test_score

    def predict(self, X, *args, **kwargs) -> pd.Series:
        """Fit the estimator according to the given training data (X, y)"""
        check_is_fitted(self)
        X = self.check_array(X)
        X_ = X[self.feature_names]
        X_.columns = [f'X_{i}' for i in range(len(self.feature_names))]
        return self.model.predict(X_)

    def predict_proba(self, X, *args, **kwargs) -> pd.Series:
        """Fit the estimator according to the given training data (X, y) and params"""
        check_is_fitted(self)
        X = self.check_array(X)
        X_ = X[self.feature_names]
        X_.columns = [f'X_{i}' for i in range(len(self.feature_names))]
        return self.model.predict_proba(X_)

    def best_params_greedy(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame = None,
            y_test: pd.Series = None,
            cv_n_splits: int = 3,
            params_grid: dict = None,
            initial_params: dict = None,
            *args,
            **kwargs,
    ):
        """find best params in a greedy style to prevent combination explosion.

        Args:
            X_train : pandas.DataFrame
                training table in dataframe.
            y_train: pandas.Series
                training target in series.
            X_test : pandas.DataFrame
                testing table in dataframe.
            y_test: pandas.Series
                testing target in series.
            cv_n_splits: int
                cross validation n(cv_n_splits) split.
            params_grid: dict
                candidate parameters grid.
            initial_params: dict
                initial parameters.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple(dict, float, float): best_params, train_score, test_score

        """
        if params_grid is None:
            params_grid = self.get_default_params_grid(self.model_class)
        best_params = {}
        train_score = 0
        test_score = 0
        if initial_params is not None:
            best_params.update(initial_params)

        for param_name, param_values in params_grid.items():
            cv_params = {param_name: param_values}
            estimator = self.model_class(**best_params)
            kfold = KFold(n_splits=cv_n_splits, shuffle=True, random_state=10)
            grid_search = RandomizedSearchCV(
                estimator, cv_params, scoring=self.scoring, n_iter=500, cv=kfold
            )
            grid_result = grid_search.fit(X_train, y_train)
            best_params.update(grid_result.best_params_)
            train_score = grid_result.best_score_
            if self.verbose:
                logger.info(f"best_score:{grid_result.best_score_}")
                logger.info(f"best_params:{grid_result.best_params_}")

        if self.verbose:
            logger.info(f"best_params:{best_params}")
        estimator = self.model_class(**best_params)
        estimator.fit(X_train, y_train)
        if X_test is not None and y_test is not None:
            test_score = estimator.score(X_test, y_test)
            if self.verbose:
                logger.info(f"test_score:{test_score}")
        return best_params, train_score, test_score

    @staticmethod
    def get_default_params_grid(estimator_class):
        """default params grid
        RandomForestRegressor: see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        XGBRegressor: see https://xgboost.readthedocs.io/en/stable/parameter.html
        CatBoostRegressor: see https://catboost.ai/en/docs/concepts/python-reference_catboostregressor

        Args:
            estimator_class : scikit learn style regressor
                one of [RandomForestRegressor, XGBRegressor, CatBoostRegressor].

        Returns:
            dict: default params grid.
        """
        default_params_grid_mapping = {
            # XGBRegressor: {
            #     "n_estimators": list(range(200, 1200, 200)),
            #     "max_depth": list(range(2, 9, 1)),
            #     "min_child_weight": np.arange(1, 7, 1),
            #     "gamma": np.arange(0.0, 0.7, 0.1),
            #     "subsample": np.arange(0.6, 1.0, 0.1),
            #     "learning_rate": [0.01, 0.05, 0.07, 0.1, 0.2],
            #     "reg_alpha": [0.05, 0.1, 1, 2, 3],
            #     "reg_lambda": [0.05, 0.1, 1, 2, 3],
            #     "colsample_bytree": np.arange(0.5, 1.0, 0.1),
            #     "colsample_bylevel": np.arange(0.5, 1.0, 0.1),
            # },
            RandomForestRegressor: {
                "n_estimators": list(range(200, 1200, 200)),
                "max_depth": list(range(2, 9, 1)),
                "min_samples_split": np.arange(1, 10, 1),
                "min_samples_leaf": np.arange(1, 10, 1),
                "min_weight_fraction_leaf": np.arange(0.0, 0.7, 0.1),
                "max_features": np.arange(1, 10, 1),
                "max_leaf_nodes": np.arange(1, 10, 1),
                "min_impurity_decrease": np.arange(0.0, 0.7, 0.1),
                "ccp_alpha": np.arange(0.0, 0.7, 0.1),
                "max_samples": np.arange(0.0, 1.0, 0.1),
            },
            LGBMRegressor: {
                'num_leaves': [20, 31, 40, 60],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [-1, 5, 7, 10],
                'min_child_samples': [10, 20, 30, 40],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.1, 0.5, 1],
            },
            # CatBoostRegressor: {
            #     "iterations": [100, 200, 500],
            #     "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
            #     "depth": np.arange(1, 10, 1),
            #     "l2_leaf_reg": np.arange(1, 10, 1),
            # },
        }
        return default_params_grid_mapping.get(estimator_class, {})


class Classification(Estimator):
    """
    class for wrapping classification estimator
    ...

    Attributes
    ----------
    model_class : class
        e.g.  sklearn.ensemble.XGBClassifier, xgboost.XGBClassifier, catboost.CatBoostClassifier
    params : dict
        params to initialize estimator_class
    scoring : str
        scoring method see https://scikit-learn.org/stable/modules/model_evaluation.html

    Methods
    -------
    fit(X, y):
        train by X, y
    """

    def __init__(
            self, model_class=None, params: dict = None, scoring: str = None, verbose=False
    ):
        if model_class is None:
            model_class = LGBMClassifier
        self.model_class = model_class
        self.params = params if params is not None else {}
        self.scoring = scoring if scoring is not None else "balanced_accuracy"
        self.scorer = get_scorer(self.scoring)
        self.verbose = verbose
        self.model = None
        self.feature_names = None
        if self.verbose:
            logger.info(f"estimator = {self.model_class}, scorer = {self.scorer}")
        self.estimator_type = EstimatorType.Classification

    def __sklearn_is_fitted__(self) -> bool:
        return self.model is not None and self.model.__sklearn_is_fitted__()

    def predict(self, X, *args, **kwargs) -> pd.Series:
        """Fit the estimator according to the given training data (X, y)"""
        check_is_fitted(self)
        X = self.check_array(X)
        X_ = X[self.feature_names]
        X_.columns = [f'X{i}' for i in range(len(self.feature_names))]
        return self.model.predict(X_)

    def predict_proba(self, X, *args, **kwargs) -> pd.Series:
        """Fit the estimator according to the given training data (X, y) and params"""
        check_is_fitted(self)
        X = self.check_array(X)
        X_ = X[self.feature_names]
        X_.columns = [f'X{i}' for i in range(len(self.feature_names))]
        return self.model.predict_proba(X_)

    def fit(self, X, y, *args, **kwargs) -> "Estimator":
        """Fit the estimator according to the given training data (X, y)"""
        return self.fit_with_params(X, y, params=self.params)

    def fit_with_params(self, X, y, params, *args, **kwargs) -> "Estimator":
        """Fit the estimator according to the given training data (X, y)"""
        X, y = self.check_X_y(X, y)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X.columns = [f'X{i}' for i in range(len(X.columns))]
        self.model = self.model_class(**params)
        self.model.fit(X, y)
        if self.verbose:
            score = self.scorer(self.model, X, y)
            logger.info(f"training score = {score}, scorer = {self.scorer}")
        return self

    def fit_with_evaluation(
            self, X_train, y_train, X_test, y_test, *args, **kwargs
    ) -> (float, float):
        """fit / train with (X_train, y_train), and evaluate with (X_test, y_test)

        Args:
            X_train : pandas.DataFrame
                training table in dataframe.
            y_train: pandas.Series
                training target in series.
            X_test : pandas.DataFrame
                testing table in dataframe.
            y_test: pandas.Series
                testing target in series.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            float: the evaluation score in [0, 1], the larger the better.
            float: the evaluation score in [0, 1], the larger the better.
        """
        self.fit(X_train, y_train)
        train_score = self.scorer(self.model, X_train, y_train)
        test_score = self.scorer(self.model, X_test, y_test)
        if self.verbose:
            logger.info(f"testing score = {test_score}, scorer = {self.scorer}")
        return train_score, test_score

    def best_params_greedy(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame = None,
            y_test: pd.Series = None,
            cv_n_splits: int = 3,
            params_grid: dict = None,
            initial_params: dict = None,
            *args,
            **kwargs,
    ):
        """find best params in a greedy style to prevent combination explosion.

        Args:
            X_train : pandas.DataFrame
                training table in dataframe.
            y_train: pandas.Series
                training target in series.
            X_test : pandas.DataFrame
                testing table in dataframe.
            y_test: pandas.Series
                testing target in series.
            cv_n_splits: int
                cross validation n(cv_n_splits) split.
            params_grid: dict
                candidate parameters grid.
            initial_params: dict
                initial parameters.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple(dict, float, float): best_params, train_score, test_score

        """
        if params_grid is None:
            params_grid = self.get_default_params_grid(self.model_class)
        best_params = {}
        train_score = 0
        test_score = 0
        if initial_params is not None:
            best_params.update(initial_params)

        for param_name, param_values in params_grid.items():
            cv_params = {param_name: param_values}
            model = self.model_class(**best_params)
            kfold = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=10)
            grid_search = RandomizedSearchCV(
                model, cv_params, scoring=self.scoring, n_iter=500, cv=kfold
            )
            grid_result = grid_search.fit(X_train, y_train)
            best_params.update(grid_result.best_params_)
            train_score = grid_result.best_score_
            if self.verbose:
                logger.info(f"best_score:{grid_result.best_score_}")
                logger.info(f"best_params:{grid_result.best_params_}")

        if self.verbose:
            logger.info(f"best_params:{best_params}")
        model = self.model_class(**best_params)
        model.fit(X_train, y_train)
        if X_test is not None and y_test is not None:
            test_score = model.score(X_test, y_test)
            if self.verbose:
                logger.info(f"test_score:{test_score}")
        return best_params, train_score, test_score

    @staticmethod
    def get_default_params_grid(estimator_class):
        """default params grid
        RandomForestClassifier: see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        XGBClassifier: see https://xgboost.readthedocs.io/en/stable/parameter.html
        CatBoostClassifier: see https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier

        Args:
            estimator_class : scikit learn style regressor
                one of [RandomForestRegressor, XGBRegressor, CatBoostRegressor].

        Returns:
            dict: default params grid.
        """
        default_params_grid_mapping = {
            # XGBClassifier: {
            #     "n_estimators": list(range(200, 1200, 200)),
            #     "max_depth": list(range(2, 9, 1)),
            #     "min_child_weight": np.arange(1, 7, 1),
            #     "gamma": np.arange(0.0, 0.7, 0.1),
            #     "subsample": np.arange(0.6, 1.0, 0.1),
            #     "learning_rate": [0.01, 0.05, 0.07, 0.1, 0.2],
            #     "reg_alpha": [0.05, 0.1, 1, 2, 3],
            #     "reg_lambda": [0.05, 0.1, 1, 2, 3],
            #     "colsample_bytree": np.arange(0.5, 1.0, 0.1),
            #     "colsample_bylevel": np.arange(0.5, 1.0, 0.1),
            # },
            RandomForestClassifier: {
                "n_estimators": list(range(200, 1200, 200)),
                "max_depth": list(range(2, 9, 1)),
                "min_samples_split": np.arange(1, 10, 1),
                "min_samples_leaf": np.arange(1, 10, 1),
                "min_weight_fraction_leaf": np.arange(0.0, 0.7, 0.1),
                "max_features": np.arange(1, 10, 1),
                "max_leaf_nodes": np.arange(1, 10, 1),
                "min_impurity_decrease": np.arange(0.0, 0.7, 0.1),
                "ccp_alpha": np.arange(0.0, 0.7, 0.1),
                "max_samples": np.arange(0.0, 1.0, 0.1),
            },
            LGBMClassifier: {
                'num_leaves': [20, 31, 40, 60],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [-1, 5, 7, 10],
                'min_child_samples': [10, 20, 30, 40],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.1, 0.5, 1],
            }
            # CatBoostClassifier: {
            #     "iterations": [100, 200, 500],
            #     "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
            #     "depth": np.arange(1, 10, 1),
            #     "l2_leaf_reg": np.arange(1, 10, 1),
            # },
        }
        return default_params_grid_mapping.get(estimator_class, {})


class ScoreCardRegression(Estimator):
    """
    class for wrapping regression scorecard estimator

    ...

    Attributes
    ----------
    model_class : scikit line regression class
        e.g。 sklearn.linear_model.LinearRegression,
        sklearn.linear_model.Ridge,
        sklearn.linear_model.HuberRegressor
    lm_params : dict
        params to initialize estimator_class
    scoring : str in ['ks', 'roc_auc']
        roc_auc scoring method see https://scikit-learn.org/stable/modules/model_evaluation.html
        ks scoring method see bibifactor.base.ks_score

    Methods
    -------
    fit(X, y):
        train by X, y
    """

    def __init__(
            self,
            model_class=None,
            lm_params: dict = None,
            scoring: str = None,
            verbose=False,
    ):
        if model_class is None:
            model_class = Ridge
        self.model_class = model_class
        if lm_params is None:
            lm_params = dict()
        self.lm_params = lm_params
        self.model = self.model_class(**self.lm_params)
        self.scorecard = None
        self.scoring = scoring if scoring is not None else "kendalltau"
        self.score_func = (
            kendalltau_score
            if self.scoring == "kendalltau"
            else get_scorer(self.scoring)._score_func
        )
        self._score_funcs = [kendalltau_score, spearmanr_score]
        self.feature_names = None
        self.verbose = verbose
        if self.verbose:
            logger.info(
                f"estimator = {self.model_class}, score_func = {self.score_func}"
            )
        self.plot_images = {}

    def fit(self, X, y, *args, **kwargs) -> "Estimator":
        """Fit the estimator according to the given training data (X, y)"""
        self.fit_with_params(X, y, lm_params=self.lm_params, *args, **kwargs)
        return self

    def fit_with_params(
            self,
            X,
            y,
            lm_params=None,
            binning_params=None,
            scorecard_params=None,
            *args,
            **kwargs,
    ) -> "Estimator":
        """Fit the estimator according to the given training data (X, y)"""
        X, y = self.check_X_y(X, y)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X.columns = [f'X{i}' for i in range(len(X.columns))]
        X_factors = self.feature_names
        if lm_params is None:
            lm_params = dict(alpha=1.0)
        if binning_params is None:
            binning_params = {}
        if scorecard_params is None:
            scorecard_params = dict(
                intercept_based=False,
                scaling_method="min_max",
                scaling_method_params={"min": 0, "max": 100},
                reverse_scorecard=False,
            )
        self.model = self.model_class(**lm_params)
        binning_process = BinningProcess(X_factors, **binning_params)
        self.scorecard = Scorecard(
            binning_process=binning_process,
            estimator=self.model,
            verbose=self.verbose,
            **scorecard_params,
        )
        self.scorecard.fit(X, y * (1 + 1e-5), show_digits=4)

        if self.verbose:
            y_pred = self.scorecard.predict(X)
            score = self.score_func(y, y_pred)
            logger.info(f"training score = {score}, score_func = {self.score_func}")
        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        """Fit the estimator according to the given training data (X, y)"""
        check_is_fitted(self)
        X = self.check_array(X)
        return self.scorecard.predict(X)

    def predict_proba(self, X, *args, **kwargs) -> pd.Series:
        """Fit the estimator according to the given training data (X, y) and params"""
        check_is_fitted(self)
        X = self.check_array(X)
        return self.scorecard.predict_proba(X)

    def fit_with_evaluation(
            self, X_train, y_train, X_test, y_test, *args, **kwargs
    ) -> float:
        """fit / train with (X_train, y_train), and evaluate with (X_test, y_test)

        Args:
            X_train : pandas.DataFrame
                training table in dataframe.
            y_train: pandas.Series
                training target in series.
            X_test : pandas.DataFrame
                testing table in dataframe.
            y_test: pandas.Series
                testing target in series.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            float: the evaluation score in [0, 1], the larger the better.
        """
        self.fit(X_train, y_train)

        y_pred = self.scorecard.predict(X_test)
        score = self.score_func(y_test, y_pred)
        if self.verbose:
            logger.info(f"testing score = {score}, score_func = {self.score_func}")
            for score_func in self._score_funcs:
                if score_func != self.score_func:
                    score_ = score_func(y_test, y_pred)
                    logger.info(f"testing score = {score_}, score_func = {score_func}")

        return score

    def best_params_greedy(
            self,
            X_train,
            y_train,
            X_test,
            y_test,
            cv_n_splits=3,
            params_grid=None,
            initial_params=None,
    ):
        pass


class ScoreCardClassification(Estimator):
    """
    class for wrapping classification scorecard estimator


    ...

    Attributes
    ----------
    lr_params : dict
        params to initialize estimator_class
    scoring : str in ['ks', 'roc_auc']
        roc_auc scoring method see https://scikit-learn.org/stable/modules/model_evaluation.html
        ks scoring method see bibifactor.base.ks_score

    Methods
    -------
    fit(X, y):
        train by X, y
    """

    def __init__(self, lr_params: dict = None, scoring: str = None, verbose=False):
        self.model_class = LogisticRegression
        if lr_params is None:
            lr_params = dict(
                solver="lbfgs",
                max_iter=2000,
                fit_intercept=True,
                tol=0.0001,
                C=0.1,
                penalty="l2",
            )
        self.lr_params = lr_params
        self.model = self.model_class(**self.lr_params)
        self.scorecard = None
        self.scoring = scoring if scoring is not None else "ks"
        self.score_func = (
            ks_score if self.scoring == "ks" else get_scorer(self.scoring)._score_func
        )
        self._score_funcs = [
            ks_score,
            roc_auc_score,
            ranking_precision_score_1_basis_points,
        ]
        self.verbose = verbose
        self.model = None
        if self.verbose:
            logger.info(
                f"estimator = {self.model_class}, score_func = {self.score_func}"
            )
        self.plot_images = {}

    def __sklearn_is_fitted__(self) -> bool:
        return self.model is not None and self.model.__sklearn_is_fitted__()

    def fit(self, X, y, *args, **kwargs) -> "Estimator":
        """Fit the estimator according to the given training data (X, y)"""
        self.fit_with_params(X, y, lr_params=self.lr_params, *args, **kwargs)
        return self

    def fit_with_params(
            self,
            X,
            y,
            lr_params=None,
            binning_params=None,
            scorecard_params=None,
            *args,
            **kwargs,
    ) -> "Estimator":
        """Fit the estimator according to the given training data (X, y)"""
        X, y = self.check_X_y(X, y)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X.columns = [f'X{i}' for i in range(len(X.columns))]
        X_factors = self.feature_names
        if lr_params is None:
            lr_params = dict(
                solver="lbfgs",
                max_iter=2000,
                fit_intercept=True,
                tol=0.0001,
                C=0.1,
                penalty="l2",
            )
        if binning_params is None:
            binning_params = {}
        if scorecard_params is None:
            scorecard_params = dict(
                intercept_based=False,
                scaling_method="pdo_odds",
                scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600},
                reverse_scorecard=False,
            )
        self.model = LogisticRegression(**lr_params)
        binning_process = BinningProcess(X_factors, **binning_params)
        self.scorecard = Scorecard(
            binning_process=binning_process,
            estimator=self.model,
            verbose=self.verbose,
            **scorecard_params,
        )
        self.scorecard.fit(X, y, show_digits=4)
        if self.verbose:
            y_pred = self.scorecard.predict_proba(X)[:, 1]
            score = self.score_func(y, y_pred)
            logger.info(f"training score = {score}, score_func = {self.score_func}")
        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        """Fit the estimator according to the given training data (X, y)"""
        check_is_fitted(self)
        X = self.check_array(X)
        return self.scorecard.predict(X)

    def predict_proba(self, X, *args, **kwargs) -> pd.Series:
        """Fit the estimator according to the given training data (X, y) and params"""
        check_is_fitted(self)
        X = self.check_array(X)
        return self.scorecard.predict_proba(X)

    def fit_with_evaluation(
            self, X_train, y_train, X_test, y_test, *args, **kwargs
    ) -> float:
        """fit / train with (X_train, y_train), and evaluate with (X_test, y_test)

        Args:
            X_train : pandas.DataFrame
                training table in dataframe.
            y_train: pandas.Series
                training target in series.
            X_test : pandas.DataFrame
                testing table in dataframe.
            y_test: pandas.Series
                testing target in series.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            float: the evaluation score in [0, 1], the larger the better.
        """
        self.fit(X_train, y_train)

        y_pred = self.scorecard.predict_proba(X_test)[:, 1]
        score = self.score_func(y_test, y_pred)
        if self.verbose:
            logger.info(f"testing score = {score}, score_func = {self.score_func}")
            for score_func in self._score_funcs:
                if score_func != self.score_func:
                    score_ = score_func(y_test, y_pred)
                    logger.info(f"testing score = {score_}, score_func = {score_func}")
        if type_of_target(y_test) == "binary":
            plot_auc_roc(y_test, y_pred, savefig=True, fname="plot_auc_roc.jpg")
            plot_ks(y_test, y_pred, savefig=True, fname="plot_ks.jpg")
            plot_precision_recall_curve(
                y_test, y_pred, savefig=True, fname="plot_precision_recall_curve.jpg"
            )
            # plot_roc_curve(y_test, y_pred, savefig=True, fname='plot_roc_curve.jpg')

        return score

    def best_params_greedy(
            self,
            X_train,
            y_train,
            X_test,
            y_test,
            cv_n_splits=3,
            params_grid=None,
            initial_params=None,
    ):
        grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}


if __name__ == "__main__":
    import pathlib
    import pandas as pd
    import numpy as np


    def split_dataframe_by_date(df, datetime_col, test_ratio=0.2):
        sorted_df = df.sort_values(datetime_col).reset_index(drop=True)
        desired_test_size = test_ratio * len(sorted_df)
        date_groups = sorted_df.groupby(datetime_col).size().reset_index(name='counts')
        date_groups_sorted = date_groups.sort_values(datetime_col, ascending=False).reset_index(drop=True)
        date_groups_sorted['cum_counts'] = date_groups_sorted['counts'].cumsum()
        closest_idx = (date_groups_sorted['cum_counts'] - desired_test_size).abs().idxmin()
        split_date = date_groups_sorted.loc[closest_idx, datetime_col]
        train = sorted_df[sorted_df[datetime_col] < split_date]
        test = sorted_df[sorted_df[datetime_col] >= split_date]
        return train, test, split_date


    data_root = pathlib.Path(r"D:\2025\新投研\非标项目\地区经济指数")
    # XyData = pd.read_excel(data_root / 'XY_data.xlsx')
    # XyData = pd.read_excel(data_root / 'XY_data_6档.xlsx')
    XyData = pd.read_excel(data_root / 'XY_data_6档_2024.xlsx')

    variable_profile_path = data_root / 'variable_profile.xlsx'
    # y_name = '更新为5档排序'
    # y_name = '运用长江观点系数模拟的排序'
    y_name = '集团敲定6档排序'
    date_name = 'date_type'
    test_ratio = 0.2


    def estimation(model_path, XyData, variable_profile_path, y_name, estimator_type,
                   date_name=None, test_ratio=0.2):
        variable_profile = read_data_file(variable_profile_path)
        feature_importances = variable_profile['feature_importance']
        feature_importances = feature_importances / sum(feature_importances)
        features = variable_profile[(feature_importances > 0.01) &
                                    (variable_profile['missing'] < 0.7) &
                                    (variable_profile['n_bins'] > 1) &
                                    (variable_profile['quality_score'] > 0)]['name'].tolist()
        X = XyData[features]
        y = XyData[y_name]
        if date_name is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=10)
        else:
            train_df, test_df, best_split_date = split_dataframe_by_date(XyData, date_name, test_ratio=test_ratio)
            train_df = train_df.iloc[np.random.permutation(len(train_df))].reset_index(drop=True)
            test_df = test_df.iloc[np.random.permutation(len(test_df))].reset_index(drop=True)
            X_train, X_test, y_train, y_test = train_df[features], test_df[features], train_df[y_name], test_df[y_name]
        train_score, test_score = estimate_and_save(X_train, y_train, X_test, y_test, estimator_type, model_path)
        return train_score, test_score


    model_path = data_root
    # estimation(model_path, XyData, variable_profile_path, y_name, EstimatorType.Regression, date_name, test_ratio)
    estimation(model_path, XyData, variable_profile_path, y_name, EstimatorType.Classification, date_name, test_ratio)
    model_all_test_y, model_test_y = load_and_predict(model_path, XyData)
    XyData['Predict_Y'] = model_test_y

    v_levels = pd.read_excel(data_root / '标尺.xlsx')
    level_dict = {k: v[0] for k, v in v_levels.to_dict().items()}
    del level_dict['长江分档分位数系数']

    model_test_y_norm = model_test_y / max(model_test_y)
    y_levels = []
    split_levels = sorted(list(level_dict.keys()))
    split_levels.append(1.1)
    for y in model_test_y_norm:
        for k, v in enumerate(split_levels):
            if y < v:
                print(y,v)
                y_levels.append(k)
                break
    from collections import Counter
    counts = Counter(y_levels)
    XyData['Predict_Level'] = model_test_y.round()
    XyData.to_excel(model_path / 'XyData_Predict_regression.xlsx')
    XyData.to_excel(model_path / 'XyData_Predict_Classification.xlsx')

    predictDF = XyData[XyData['date_type'] == 2023][['regname','Predict_Y']]
    trueDF = XyData[XyData['date_type'] == 2024][['regname',y_name]]
    compDF = trueDF.merge(predictDF, on=['regname'], how='inner')
    accuracy_score(compDF['Predict_Y'], compDF[y_name])
    from sklearn.metrics import f1_score
    f1_score(compDF['Predict_Y'], compDF[y_name], average='weighted')
