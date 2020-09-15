import os
import logging

from .xgboost_utils import OheFeatureGenerator
from .hyperparameters.parameters import get_param_baseline
from ..abstract.abstract_model import AbstractModel
from ...constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS, PROBLEM_TYPES_CLASSIFICATION

logger = logging.getLogger(__name__)


class XGBoostModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ohe_generator = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type, num_classes=self.num_classes)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO: Enable HPO for XGBoost
    def _get_default_searchspace(self):
        spaces = {}
        return spaces

    def preprocess(self, X, is_train=False):
        X = super().preprocess(X=X)

        if self._ohe_generator is None:
            self._ohe_generator = OheFeatureGenerator()

        if is_train:
            self._ohe_generator.fit(X)

        X = self._ohe_generator.transform(X)

        return X

    def _fit(self, X_train, y_train, **kwargs):
        from xgboost import XGBClassifier, XGBRegressor
        model_type = XGBClassifier if self.problem_type in PROBLEM_TYPES_CLASSIFICATION else XGBRegressor

        X_train = self.preprocess(X_train, is_train=True)
        params = self.params.copy()
        self.model = model_type(**params)
        self.model.fit(X_train, y_train)

    def get_model_feature_importance(self):
        original_feature_names: list = self._ohe_generator.get_original_feature_names()
        feature_names = self._ohe_generator.get_feature_names()
        importances = self.model.feature_importances_.tolist()

        importance_dict = {}
        for original_feature in original_feature_names:
            importance_dict[original_feature] = 0
            for feature, value in zip(feature_names, importances):
                if feature in self._ohe_generator.othercols:
                    importance_dict[feature] = value
                else:
                    feature = '_'.join(feature.split('_')[:-1])
                    if feature == original_feature:
                        importance_dict[feature] += value

        return importance_dict
