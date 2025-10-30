from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression


class PLSTransformer(BaseEstimator, TransformerMixin):
    """
    Supervised dimensionality reduction using PLSRegression under the hood.
    - Accepts class labels y (integers) and internally one-hot encodes them.
    - Returns X scores (n_samples, n_components).
    """

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self._pls: Optional[PLSRegression] = None
        self._classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        # derive classes and one-hot encode
        classes = np.unique(y)
        self._classes_ = classes
        class_to_index = {c: i for i, c in enumerate(classes)}
        Y = np.zeros((y.shape[0], classes.shape[0]), dtype=np.float64)
        for i, label in enumerate(y):
            Y[i, class_to_index[label]] = 1.0

        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X, Y)
        self._pls = pls
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pls is None:
            raise RuntimeError("PLSTransformer not fitted")
        X = np.asarray(X)
        X_scores = self._pls.transform(X)
        # sklearn returns X_scores only when Y is not provided in transform
        # Ensure 2D array
        return np.asarray(X_scores)


