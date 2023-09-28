from typing import Union
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

from .sketches import GaussianSketchFactory, OrthogonalSketchFactory, CountSketchFactory, SubsampledFourierSketchFactory
from .sketches import SketchSizeParams


class SVDRidge(RegressorMixin, BaseEstimator):
    """Ridge regression via SVD"""

    def __init__(self, lamda: float = 1) -> None:

        self.lamda = lamda
        self.coef_ = None

    def fit(self, X, y):

        self.n_ = X.shape[0]

        self.X_mean_ = np.mean(X, axis=0)
        X = X - self.X_mean_[None, :]
        self.y_mean_ = np.mean(y)
        y = y - self.y_mean_

        U, self.s_, Vh = np.linalg.svd(X, full_matrices=False)
        self.Uy_ = U.conj().T @ y
        self.V_ = Vh.conj().T

        self.coef_ = self.V_ @ (self.s_ / (self.s_ ** 2 + self.n_ * self.lamda) * self.Uy_)

        return self
    
    def predict(self, X):
            
        check_is_fitted(self, ('coef_', 'X_mean_', 'y_mean_'))

        return (X - self.X_mean_[None, :]) @ self.coef_ + self.y_mean_
    
    def refit(self, lamda: float = 1):

        check_is_fitted(self, ('V_', 's_', 'Uy_', 'n_'))

        self.lamda = lamda

        self.coef_ = self.V_ @ (self.s_ / (self.s_ ** 2 + self.n_ * self.lamda) * self.Uy_)

        return self


class PreSketchedEnsembleRidge(RegressorMixin, BaseEstimator):
    """Ridge regression ensemble with pre-sketched data."""

    def __init__(self, n_ens: int = 10, lamda: float = 1, which_ridge: str = 'svd') -> None:

        self.n_ens = n_ens
        self.lamda = lamda
        self.which_ridge = which_ridge

        self.ensemble_ = None

    def fit(self, X, y):

        n, p = X.shape
        q = p // self.n_ens
        
        if q * self.n_ens != p:
            raise ValueError('Must have an equal number of features for each ensemble member (q * n_ens = p).')
        
        if self.which_ridge == 'svd':
            ridge_cls = SVDRidge
        else:
            ridge_cls = lambda lamda: Ridge(alpha=n * lamda)

        self.ensemble_ = [ridge_cls(self.lamda).fit(X[:, i * q:(i + 1) * q], y) for i in range(self.n_ens)]

        return self
    
    def predict(self, X):

        check_is_fitted(self, 'ensemble_')

        n, p = X.shape
        q = p // self.n_ens
        
        if q * self.n_ens != p:
            raise ValueError('Must have an equal number of features for each ensemble member (q * n_ens = p).')
        
        return np.mean([m.predict(X[:, i * q:(i + 1) * q]) for i, m in enumerate(self.ensemble_)], axis=0)
    
    def refit(self, lamda):

        if self.which_ridge != 'svd':
            raise ValueError('Can only refit with SVD ridge.')

        check_is_fitted(self, 'ensemble_')

        self.lamda = lamda

        for m in self.ensemble_:
            m.refit(lamda)

        return self

    def gcv_div(self):

        if self.which_ridge != 'svd':
            raise ValueError('Can only compute GCV correction with SVD ridge.')

        check_is_fitted(self, 'ensemble_')

        return np.mean([np.sum(m.s_ ** 2 / (m.s_ ** 2 + m.n_ * self.lamda)) / m.n_ for m in self.ensemble_])
    

class EnsemblePreSketchTransform(TransformerMixin):

    def __init__(self, n_ens: int = 10, q: Union[int, str] = 'auto', sketch: str = 'dct', rng: np.random.Generator = None) -> None:

        self.n_ens = n_ens
        self.q = q
        self.sketch = sketch
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.sketches_ = None

    def fit(self, X, y=None):

        n, p = X.shape

        if self.sketch == 'gaussian':
            sketch_factory = GaussianSketchFactory(p, rng=self.rng)
        elif self.sketch == 'orthogonal':
            sketch_factory = OrthogonalSketchFactory(p, rng=self.rng)
        elif self.sketch == 'countsketch':
            sketch_factory = CountSketchFactory(p, rng=self.rng)
        elif self.sketch == 'dct':
            sketch_factory = SubsampledFourierSketchFactory(p, transform='dct', rng=self.rng)   
        
        if self.q == 'auto':
            q = p // self.n_ens
        else:   
            q = self.q
        
        self.sketches_ = [sketch_factory(SketchSizeParams(q)) for _ in range(self.n_ens)]
    
        return self
    
    def transform(self, X, y=None):

        check_is_fitted(self, 'sketches_')

        n, p = X.shape

        return np.concatenate([(sketch @ X.conj().T).conj().T for sketch in self.sketches_], axis=1)


