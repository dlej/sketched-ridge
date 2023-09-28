from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy.fft import fft, ifft, dct, idct
from scipy import sparse
from scipy.sparse.linalg import LinearOperator as spLinearOperator


""" All sketches are LinearOperators designed to be used as follows:

    `y = A @ x`

    where `y` is the sketched vector, `A` is the sketching operator, and `x` is
    the original vector. This is the transpose of the sketching matrices 
    considered in the text of our paper.
"""


def get_count_sketch_matrix(b, t, m, normed=False, rng=None, dtype=None):
    """ notation from Charikar, Moses, Kevin Chen, and Martin Farach-Colton.
    "Finding frequent items in data streams." International Colloquium on
    Automata, Languages, and Programming. 2002.
    
    build the sketching matrix for Count Sketch for `m` items with `t` hash
    functions mapping onto `{1, ..., b}`.
    """

    if rng is None:
        rng = np.random.default_rng()

    data = rng.choice(np.asarray([-1, 1]), size=t * m, replace=True)
    if normed:
        data = data / np.sqrt(t)

    hash = rng.choice(b, size=(m, t), replace=True)
    row_ind = (hash + b * np.arange(t)[None, :]).ravel()
    col_ind = np.repeat(np.arange(m), t)

    return sparse.csc_matrix((data, (row_ind, col_ind)), shape=(b * t, m), dtype=dtype)


def get_subsample_matrix(q, p, normed=False, rng=None, dtype=None):

    if rng is None:
        rng = np.random.default_rng()
    
    data = np.ones(q)
    if normed:
        data = data * np.sqrt(p / q)

    row_ind = np.arange(q)
    col_ind = rng.choice(p, q, replace=False)

    return sparse.csc_matrix((data, (row_ind, col_ind)), shape=(q, p), dtype=dtype)


def get_permutation_matrix(p, rng=None, dtype=None):

    if rng is None:
        rng = np.random.default_rng()

    data = np.ones(p, dtype=dtype)

    row_ind = np.arange(p)
    col_ind = rng.permutation(p)

    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(p,  p), dtype=dtype)


def any_to_array(x, dtype=None):

    if isinstance(x, np.ndarray):
        return x

    elif isinstance(x, sparse.spmatrix):
        return x.toarray()

    elif isinstance(x, spLinearOperator):
        return x.toarray()

    else:
        return np.asarray(x, dtype=dtype)
    

# fix issues with scipy.sparse.linalg.LinearOperator
class LinearOperator(spLinearOperator):

    def dot(self, x):

        if isinstance(x, sparse.spmatrix):
            y = self._matmat(x)
        else:
            y = super().dot(x)
        
        return any_to_array(y)
    
    def todense(self):
        return self @ np.eye(self.shape[1])


class MatMatMult(LinearOperator):

    def __init__(self, A):
        super().__init__(shape=A.shape, dtype=A.dtype)
        self.A = A
        self.explicit = False
    
    def _matmat(self, X):
        return self.A @ X
    
    def _rmatmat(self, X):
        return self.A.conj().T @ X
    
    def _matvec(self, x):
        return self.A @ x
    
    def _rmatvec(self, x):
        return self.A.conj().T @ x
    

class CountSketch(MatMatMult):

    def __init__(self, b, t, m, normed=True, rng=None, dtype=float):

        A = get_count_sketch_matrix(b, t, m, normed=normed, rng=rng, dtype=dtype)
        super().__init__(A)


class SubsampleSketch(MatMatMult):

    def __init__(self, q, p, normed=False, rng=None, dtype=float):

        A = get_subsample_matrix(q, p, normed=normed, rng=rng, dtype=dtype)
        super().__init__(A)


class GaussianSketch(MatMatMult):
    
    def __init__(self, q, p, normed=True, rng=None, dtype=float):

        if rng is None:
            rng = np.random.default_rng()
        
        A = rng.normal(scale=1 / np.sqrt(p), size=(q, p)).astype(dtype)

        if normed:
            A *= np.sqrt(p / q)

        super().__init__(A)


class OrthogonalSketch(MatMatMult):
    
    def __init__(self, q, p, normed=True, rng=None, dtype=float):

        if q >= p:
            Q = np.eye(p, dtype=dtype)

        else:
            if rng is None:
                rng = np.random.default_rng()
            
            A = rng.normal(size=(p, q)).astype(dtype)
            Q, _ = np.linalg.qr(A)

            if normed:
                Q *= np.sqrt(p / q)

        super().__init__(Q.T)


class SubsampledFourierSketch(LinearOperator):

    def __init__(self, q, p, transform='fft', normed=True, perm=False, rng=None, dtype=float):

        if rng is None:
            rng = np.random.default_rng()
        
        self.p = p
        self.transform = transform

        self.P = get_subsample_matrix(q, p, normed=normed, rng=rng, dtype=dtype)
        self.D = sparse.spdiags(rng.choice([-1, 1], size=(1, p), replace=True), [0], p, p)

        if perm:
            self.D = get_permutation_matrix(p) @ self.D

        super().__init__(dtype, (q, p))
    
    def _matmat(self, X):

        Z = any_to_array(self.D @ X)
        if self.transform == 'fft':
            Z = fft(Z, axis=0, norm='ortho')
        elif self.transform == 'dct':
            Z = dct(Z, axis=0, norm='ortho')
        Z = self.P @ Z

        return Z
    
    def _adjoint(self):
        return FJLTranspose(self)


class FJLTranspose(LinearOperator):

    def __init__(self, fjl):

        self.fjl = fjl

        super().__init__(fjl.dtype, fjl.shape[::-1])
    
    def _matmat(self, X):

        Z = any_to_array(self.fjl.P.T @ X)
        if self.fjl.transform == 'fft':
            Z = ifft(Z, axis=0, norm='ortho')
        elif self.fjl.transform == 'dct':
            Z = idct(Z, axis=0, norm='ortho')
        Z = self.fjl.D.T @ Z

        return Z
    
    def _adjoint(self):
        return self.fjl


class SketchParams(object):
    pass


@dataclass(frozen=True, eq=True)
class SketchSizeParams(SketchParams):
    q: int


class SketchFactory(ABC):

    def __init__(self, p) -> None:
        self.p = p
    
    @abstractmethod
    def __call__(self, sketch_params: SketchParams, **kwargs):
        pass


class Identity(LinearOperator):

    def __init__(self, p) -> None:
        super().__init__(shape=(p, p), dtype=float)
    
    def _matmat(self, X):
        return X
    
    def _adjoint(self):
        return self


class IdentityFactory(SketchFactory):

    def __call__(self, *args, **kwargs):
        return Identity(self.p)


class GaussianSketchFactory(SketchFactory):

    def __init__(self, p, normed=True, rng=None, dtype=float) -> None:
        super().__init__(p)
        self.normed = normed
        self.rng = rng
        self.dtype = dtype
    
    def __call__(self, sketch_size_params: SketchSizeParams):
        return GaussianSketch(sketch_size_params.q, self.p, normed=self.normed, rng=self.rng, dtype=self.dtype)


class OrthogonalSketchFactory(SketchFactory):
    
        def __init__(self, p, normed=True, rng=None, dtype=float) -> None:
            super().__init__(p)
            self.normed = normed
            self.rng = rng
            self.dtype = dtype
        
        def __call__(self, sketch_size_params: SketchSizeParams):
            return OrthogonalSketch(sketch_size_params.q, self.p, normed=self.normed, rng=self.rng, dtype=self.dtype)
        

class CountSketchFactory(SketchFactory):

    def __init__(self, p, normed=True, rng=None, dtype=float) -> None:
        super().__init__(p)
        self.normed = normed
        self.rng = rng
        self.dtype = dtype
    
    def __call__(self, sketch_size_params: SketchSizeParams):
        t = int(10 * max(1, np.log(self.p)))
        b = int(max(1, sketch_size_params.q / t))
        return CountSketch(b, t, self.p, normed=self.normed, rng=self.rng, dtype=self.dtype)


class SubsampledFourierSketchFactory(SketchFactory):

    def __init__(self, p, transform='fft', normed=True, perm=False, rng=None, dtype=float) -> None:
        super().__init__(p)
        self.transform = transform
        self.normed = normed
        self.perm = perm
        self.rng = rng
        self.dtype = dtype
    
    def __call__(self, sketch_size_params: SketchSizeParams):
        return SubsampledFourierSketch(sketch_size_params.q, self.p, transform=self.transform, normed=self.normed, perm=self.perm, rng=self.rng, dtype=self.dtype)