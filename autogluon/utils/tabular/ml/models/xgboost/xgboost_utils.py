from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class OheFeatureGenerator(BaseEstimator, TransformerMixin):
    null_category_str = '__NaN__'

    def __init__(self):
        self._feature_map = {}  # key: feature_name, value: feature_type
        self.cat_cols = []
        self.other_cols = []
        self.ohe_encs = None
        self.labels = None

    def fit(self, X, y=None):
        self.cat_cols = list(X.select_dtypes(include='category').columns)
        self.other_cols = list(X.select_dtypes(exclude='category').columns)

        self.ohe_encs = {f: OneHotEncoder(handle_unknown='ignore') for f in self.cat_cols}
        self.labels = {}

        for c in self.cat_cols:
            self.ohe_encs[c].fit(self._normalize(X[c]))
            self.labels[c] = self.ohe_encs[c].categories_

        # Update feature map ({name: type})
        for k, v in self.labels.items():
            for f in k + '_' + v[0]:
                self._feature_map[f] = 'i'  # one-hot encoding data type is boolean

        for c in self.other_cols:
            if X[c].dtypes == int:
                self._feature_map[c] = 'int'
            else:
                self._feature_map[c] = 'float'

        return self

    def transform(self, X, y=None):
        X_list = [self.ohe_encs[c].transform(self._normalize(X[c])) for c in self.cat_cols]
        X_list.append(X[self.other_cols])

        return hstack(X_list, format="csr")

    def _normalize(self, col):
        return col.astype(str).fillna(self.null_category_str).values.reshape(-1, 1)

    def get_feature_names(self):
        return list(self._feature_map.keys())

    def get_feature_types(self):
        return list(self._feature_map.values())
