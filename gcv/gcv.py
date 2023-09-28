from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Hashable, Optional, Type, Union

import numpy as np
from pylops import LinearOperator, Identity, MatrixMult
from pylops.optimization.basic import cgls
from pylops.utils.typing import NDArray

from .sketches import SketchFactory, IdentityFactory, SketchParams


class NoRegressionCoefficientsError(Exception):
    """Exception for not having regression coefficients."""

    pass


class RegressionParams(object):
    pass


@dataclass(frozen=True, eq=True)
class SketchedRidgeParams(RegressionParams):
    lamda: float
    sketch_params: SketchParams = None


class Regressor(ABC):
    """Abstract class for a regressor."""

    def __init__(self, X: NDArray, y: NDArray) -> None:
        """Initialize the class.

        parameters
        ----------
        X : NDArray
            The training data matrix.
        y : NDArray
            The training labels.
        """

        self.n, self.p = X.shape
        self.X = X
        self.y = y

    @abstractmethod
    def div(self, params: RegressionParams) -> float:
        """Compute the divergence of the prediction.

        parameters
        ----------
        params : ParamsType
            The parameters of the regressor.

        returns
        -------
        float
            The divergence of the prediction.
        """

        pass

    @abstractmethod
    def y_hat(self, params: RegressionParams) -> NDArray:
        """Compute the prediction on training points.

        parameters
        ----------
        params : ParamsType
            The parameters of the regressor.

        returns
        -------
        NDArray
            The prediction on training points.
        """

        pass

    @abstractmethod
    def beta(self, params: RegressionParams) -> NDArray:
        """Compute the regression coefficients.

        parameters
        ----------
        params : ParamsType
            The parameters of the regressor.

        returns
        -------
        NDArray
            The regression coefficients.
        """

        pass

    @abstractmethod
    def predict(
        self, X: Union[NDArray, LinearOperator], params: RegressionParams
    ) -> NDArray:
        """Compute the prediction on test points.

        parameters
        ----------
        X : Union[NDArray, LinearOperator]
            The test data matrix.
        params : ParamsType
            The parameters of the regressor.

        returns
        -------
        NDArray
            The prediction on test points.
        """

        pass


class SketchedSVDRidge(Regressor):
    """Class for computing ridge regression for skethed data."""

    def __init__(
        self, X: NDArray, y: NDArray, sketch_factory: Optional[SketchFactory] = None
    ) -> None:
        """Initialize the class.

        parameters
        ----------
        X : NDArray
            The training data matrix.
        y : NDArray
            The training labels.
        sketch_factory : Optional[SketchFactory]
            The sketch factory.
        """

        super().__init__(X, y)

        if sketch_factory is None:
            sketch_factory = IdentityFactory(self.p)

        self.sketch_factory = sketch_factory
        if self.p != self.sketch_factory.p:
            raise ValueError("The sketch factory dimension must match that of X.")

        self.sketch_params = None

    def update_sketch(self, sketch_params: SketchParams) -> None:
        """Update the sketch.

        parameters
        ----------
        sketch_params : ParamsType
            The parameters of the sketch.
        """

        if self.sketch_params == sketch_params:
            return
        self.sketch_params = sketch_params

        self.Sh = self.sketch_factory(sketch_params)
        self.S = self.Sh.adjoint()

        self.XS = (self.Sh @ self.X.conj().T).conj().T

        self.U, self.s, self.Vh = np.linalg.svd(self.XS, full_matrices=False)
        self.XSV = self.XS @ self.Vh.conj().T
        self.Uy = self.U.conj().T @ self.y

        # thin SVD
        nnz = np.count_nonzero(self.s)
        self.U = self.U[:, :nnz]
        self.s = self.s[:nnz]
        self.Vh = self.Vh[:nnz, :]

    def betaS(self, params: SketchedRidgeParams) -> NDArray:
        """Compute the ridge regression coefficients.

        parameters
        ----------
        params : SketchedRidgeParams
            The regularization and sketch parameters.

        returns
        -------
        NDArray
            The ridge regression coefficients.
        """

        self.update_sketch(params.sketch_params)

        return self.Vh.conj().T @ (
            self.s / (self.s**2 + self.n * params.lamda) * self.Uy
        )

    def beta(self, params: SketchedRidgeParams) -> NDArray:
        """Compute the ridge regression coefficients.

        parameters
        ----------
        params : SketchedRidgeParams
            The regularization and sketch parameters.

        returns
        -------
        NDArray
            The ridge regression coefficients.
        """

        self.update_sketch(params.sketch_params)

        return self.S @ self.betaS(params)

    def div(self, params: SketchedRidgeParams) -> float:
        """Compute the divergence of the ridge regression prediction.

        parameters
        ----------
        params : SketchedRidgeParams
            The regularization and sketch parameters.

        returns
        -------
        float
            The divergence of the ridge regression prediction.
        """

        self.update_sketch(params.sketch_params)

        return np.sum(self.s**2 / (self.s**2 + self.n * params.lamda))

    def y_hat(self, params: SketchedRidgeParams) -> NDArray:
        """Compute the ridge regression prediction on training points.

        parameters
        ----------
        params : SketchedRidgeParams
            The regularization and sketch parameters.

        returns
        -------
        NDArray
            The ridge regression prediction on training points.
        """

        return self.predict(self.X)

    def predict(
        self, X: Union[NDArray, LinearOperator], params: SketchedRidgeParams
    ) -> NDArray:
        """Compute the ridge regression prediction on test points.

        parameters
        ----------
        X : Union[NDArray, LinearOperator]
            The test data matrix.
        params : SketchedRidgeParams
            The regularization and sketch parameters.

        returns
        -------
        NDArray
            The ridge regression prediction on test points.
        """

        self.update_sketch(params.sketch_params)

        return X @ self.beta(params)


class SketchedSolveRidge(Regressor):
    """Class for computing ridge regression for skethed data using linear solve."""

    def __init__(
        self, X: NDArray, y: NDArray, sketch_factory: Optional[SketchFactory] = None
    ) -> None:
        """Initialize the class.

        parameters
        ----------
        X : NDArray
            The training data matrix.
        y : NDArray
            The training labels.
        sketch_factory : SketchFactory
            The sketch factory.
        rng : Optional[np.random.Generator]
            The random number generator.
        max_iter : int
            The maximum number of iterations for the linear solve.
        """

        super().__init__(X, y)

        if sketch_factory is None:
            sketch_factory = IdentityFactory(self.p)

        self.sketch_factory = sketch_factory
        if self.p != self.sketch_factory.p:
            raise ValueError("The sketch factory dimension must match that of X.")

        self.sketch_params = None
        self.beta_params = None
        self._beta = None
        self.div_params = None
        self._div = None

    def update_sketch(self, sketch_params: SketchParams) -> None:
        """Update the sketch.

        parameters
        ----------
        sketch_params : ParamsType
            The parameters of the sketch.
        """

        if self.sketch_params == sketch_params:
            return
        self.sketch_params = sketch_params

        self.Sh = self.sketch_factory(sketch_params)
        self.S = self.Sh.adjoint()

        self.XS = (self.Sh @ self.X.conj().T).conj().T
        self.SXXS = self.XS.conj().T @ self.XS
        self.SXy = self.XS.conj().T @ self.y

        self.s2 = np.linalg.eigvalsh(self.SXXS)

    def betaS(self, params: SketchedRidgeParams) -> NDArray:
        """Compute the ridge regression coefficients.

        parameters
        ----------
        params : SketchedRidgeParams
            The regularization and sketch parameters.

        returns
        -------
        NDArray
            The ridge regression coefficients.
        """

        self.update_sketch(params.sketch_params)

        return np.linalg.solve(
            self.SXXS + self.n * params.lamda * np.eye(self.SXXS.shape[0]), self.SXy
        )

    def beta(self, params: SketchedRidgeParams) -> NDArray:
        """Compute the ridge regression coefficients.

        parameters
        ----------
        params : SketchedRidgeParams
            The regularization and sketch parameters.

        returns
        -------
        NDArray
            The ridge regression coefficients.
        """

        self.update_sketch(params.sketch_params)

        if params != self.beta_params:
            self._beta = self.S @ self.betaS(params)
            self.beta_params = params

        return self._beta

    def div(self, params: SketchedRidgeParams) -> float:
        """Estimate the divergence of the ridge regression prediction.

        parameters
        ----------
        params : SketchedRidgeParams
            The regularization and sketch parameters.

        returns
        -------
        float
            The divergence of the ridge regression prediction.
        """

        self.update_sketch(params.sketch_params)

        if params != self.div_params:
            self._div = np.sum(self.s2 / (self.s2 + self.n * params.lamda))
            self.div_params = params

            # SXXS_reg = self.SXXS + self.n * params.lamda * Identity(self.SXXS.shape[0])
            # if self.n_div is None:
            #     SXXS = self.SXXS.todense()
            #     SXXS_reg = SXXS_reg.todense()
            #     self._div = np.trace(np.linalg.solve(SXXS_reg, SXXS))
            # else:
            #     self._div = 0
            #     U, _ = np.linalg.qr(self.rng.normal(size=(self.n, self.n_div)))
            #     SXU = self.XS.adjoint() @ U
            #     for i in range(self.n_div):
            #         self._div += SXU[:, i] @ cgls(SXXS_reg, SXU[:, i], damp=0, niter=self.max_iter)[0]
            #     self._div *= self.n / self.n_div

            # self.div_params = params

        return self._div

    def y_hat(self, params: SketchedRidgeParams) -> NDArray:
        """Compute the ridge regression prediction on training points.

        parameters
        ----------
        params : SketchedRidgeParams
            The regularization and sketch parameters.

        returns
        -------
        NDArray
            The ridge regression prediction on training points.
        """

        self.update_sketch(params.sketch_params)

        return self.X @ self.beta(params)

    def predict(self, X: NDArray, params: SketchedRidgeParams) -> NDArray:
        """Compute the ridge regression prediction on test points.

        parameters
        ----------
        X : NDArray
            The test data matrix.
        params : SketchedRidgeParams
            The regularization and sketch parameters.

        returns
        -------
        NDArray
            The ridge regression prediction on test points.
        """

        self.update_sketch(params.sketch_params)

        return X @ self.beta(params)


class EnsembleGCV(object):
    """Class for computing the GCV estimate of the mean squared error for an ensemble of regressors."""

    def __init__(self, regressors: Iterable[Regressor]) -> None:
        """Initialize the class.

        parameters
        ----------
        regressors : Iterable[Regressor]
            The ensemble of regressors.
        """

        self.regressors = regressors
        self.X = self.regressors[0].X
        self.y = self.regressors[0].y
        self.n = self.regressors[0].n

        self.beta_params = None
        self._beta = None
        self.div_params = None
        self._div = None

        # check that all regressors have the same training data
        # for regressor in self.regressors:
        #     try:
        #         assert self.X is regressor.X or np.allclose(self.X, regressor.X)
        #         assert self.y is regressor.y or np.allclose(self.y, regressor.y)
        #     except AssertionError:
        #         raise ValueError("All regressors must have the same training data.")

    def beta(self, params: RegressionParams) -> NDArray:
        """Compute the regression coefficients.

        parameters
        ----------
        params : RegressionParams
            The regressor parameters.

        returns
        -------
        NDArray
            The regression coefficients.
        """

        # check that all regressors have coefficients
        for regressor in self.regressors:
            if not hasattr(regressor, "beta"):
                raise NoRegressionCoefficientsError(
                    "All regressors must have regression coefficients."
                )

        if params != self.beta_params:
            self.beta_params = params
            self._beta = np.mean(
                np.stack(
                    [predictor.beta(params) for predictor in self.regressors], axis=0
                ),
                axis=0,
            )

        return self._beta

    def div(self, params: RegressionParams) -> float:
        """Compute the divergence of the prediction.

        parameters
        ----------
        params : RegressionParams
            The regressor parameters.

        returns
        -------
        float
            The divergence of the prediction.
        """

        if params != self.div_params:
            self.div_params = params
            self._div = np.mean(
                [predictor.div(params) for predictor in self.regressors]
            )

        return self._div

    def y_hat(self, params: RegressionParams) -> NDArray:
        """Compute the prediction on training points.

        parameters
        ----------
        params : RegressionParams
            The regressor parameters.

        returns
        -------
        NDArray
            The prediction on training points.
        """

        return self.predict(self.X, params)

    def predict(
        self, X: Union[NDArray, LinearOperator], params: RegressionParams
    ) -> NDArray:
        """Compute the prediction on test points.

        parameters
        ----------
        X : Union[NDArray, LinearOperator]
            The test data matrix.
        params : RegressionParams
            The regressor parameters.

        returns
        -------
        NDArray
            The prediction on test points.
        """

        # use beta if available
        try:
            return X @ self.beta(params)
        except NoRegressionCoefficientsError:
            return np.mean(
                np.stack(
                    [predictor.predict(X, params) for predictor in self.regressors],
                    axis=0,
                ),
                axis=0,
            )

    def gcv_correction(self, params: RegressionParams) -> float:
        """Compute the GCV correction term.

        parameters
        ----------
        params : RegressionParams
            The regressor parameters.

        returns
        -------
        float
            The GCV correction term.
        """

        return 1 - self.div(params) / self.n

    def eval_train(self, params: RegressionParams) -> float:
        """Evaluate the training error.

        parameters
        ----------
        params : RegressionParams
            The regressor parameters.

        returns
        -------
        float
            The training error.
        """

        return np.linalg.norm(self.y - self.y_hat(params), 2) ** 2 / self.n

    def eval_train_path(self, params_path: Iterable[RegressionParams]) -> NDArray:
        """Evaluate the training error over a path of parameters.

        parameters
        ----------
        params_path : Iterable[ParamsType]
            The regressor parameters.

        returns
        -------
        NDArray
            The training error.
        """

        return np.array([self.eval_train(params) for params in params_path])

    def eval_gcv(self, params: RegressionParams) -> float:
        """Compute the GCV estimate of the mean squared error.

        parameters
        ----------
        params : RegressionParams
            The regressor parameters.

        returns
        -------
        float
            The GCV estimate of the mean squared error.
        """

        denom = self.gcv_correction(params) ** 2

        return self.eval_train(params) / denom

    def eval_gcv_path(self, params_path: Iterable[RegressionParams]) -> NDArray:
        """Evaluate the GCV estimate of the mean squared error over a path of parameters.

        parameters
        ----------
        params_path : Iterable[ParamsType]
            The regressor parameters.

        returns
        -------
        NDArray
            The GCV estimates of the mean squared error.
        """

        return self.eval_train_path(params_path) / np.array(
            [self.gcv_correction(params) ** 2 for params in params_path]
        )

    def eval_test(self, X: NDArray, y: NDArray, params: RegressionParams) -> float:
        """Evaluate the test error.

        parameters
        ----------
        X : NDArray
            The test data matrix.
        y : NDArray
            The test labels.
        params : ParamsType
            The regressor parameters.

        returns
        -------
        float
            The test error.
        """

        n = X.shape[0]

        return np.linalg.norm(y - self.predict(X, params), 2) ** 2 / n

    def eval_test_path(
        self, X: NDArray, y: NDArray, params_path: Iterable[RegressionParams]
    ) -> NDArray:
        """Evaluate the test error over a path of parameters.

        parameters
        ----------
        X : NDArray
            The test data matrix.
        y : NDArray
            The test labels.
        params_path : Iterable[ParamsType]
            The regressor parameters.

        returns
        -------
        NDArray
            The test error.
        """

        return np.array([self.eval_test(X, y, params) for params in params_path])

    def eval_cond_gen(
        self, Sigma: NDArray, beta: NDArray, sigma: float, params: RegressionParams
    ) -> float:
        """Evaluate the ridge regression generalization error conditioned on X, y.

        parameters
        ----------
        Sigma : NDArray
            The covariance matrix of the data.
        beta : NDArray
            The regression coefficients.
        sigma : float
            The noise level.
        params : ParamsType
            The regressor parameters.

        returns
        -------
        float
            The ridge regression generalization error.
        """

        return self.eval_cond_gen_path(Sigma, beta, sigma, [params])[0]

    def eval_cond_gen_path(
        self,
        Sigma: Union[NDArray, LinearOperator],
        beta: NDArray,
        sigma: float,
        params_path: Iterable[RegressionParams],
    ) -> NDArray:
        """Evaluate the ridge regression generalization error conditioned on X, y over a path of regularization parameters.

        parameters
        ----------
        Sigma : Union[NDArray, LinearOperator]
            The covariance matrix of the data.
        beta : NDArray
            The regression coefficients.
        sigma : float
            The noise level.
        params_path : Iterable[ParamsType]
            The regressor parameters.

        returns
        -------
        NDArray
            The ridge regression generalization errors.
        """

        gens = np.zeros(len(params_path))
        for params_idx, params in enumerate(params_path):
            beta_err = beta - self.beta(params)
            gens[params_idx] = beta_err @ Sigma @ beta_err + sigma**2

        return gens

    def __call__(self, params: RegressionParams) -> float:
        """Convenience function for eval_gcv."""

        return self.eval_gcv(params)

    def __getitem__(self, idx: Union[int, slice]) -> Regressor:
        """Obtain a sub-ensemble."""

        if isinstance(idx, int):
            return EnsembleGCV([self.regressors[idx]])
        elif isinstance(idx, slice):
            return EnsembleGCV(self.regressors[idx])
        else:
            raise TypeError("Must index with int or slice.")

    def __len__(self) -> int:
        """Number of regressors."""

        return len(self.regressors)

    def __repr__(self) -> str:
        """Representation of the object."""

        return f"Ensemble of {len(self)} regressors"
