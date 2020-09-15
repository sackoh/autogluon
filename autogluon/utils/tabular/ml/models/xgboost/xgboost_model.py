import os
import logging
from ..abstract.abstract_model import AbstractModel
from ...constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS, PROBLEM_TYPES_CLASSIFICATION

logger = logging.getLogger(__name__)


class XGBoostModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_default_params(self):
        default_params = {
            'n_estimators': 1000,
            'n_jobs': -1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO: Enable HPO for XGBoost
    def _get_default_searchspace(self):
        spaces = {}
        return spaces

    def preprocess(self, X, is_train=False):
        X = super().preprocess(X)
        return X

    def _fit(self, X_train, y_train, **kwargs):
        from xgboost import XGBClassifier, XGBRegressor
        model_type = XGBClassifier if self.problem_type in PROBLEM_TYPES_CLASSIFICATION else XGBRegressor

        X_train = self.preprocess(X_train, is_train=True)
        params = self.params.copy()
        self.model = model_type(**params)
        self.model.fit(X_train, y_train)
