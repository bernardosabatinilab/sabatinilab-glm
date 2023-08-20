from typing import DefaultDict, List, TypeVar, Optional, Union, List, Tuple, Dict
import numpy as np
import pandas as pd
import scipy.sparse

class Vector:

    def __init__(
            self,
            nrows: int = None,
            filter_indices: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
            values: Optional[np.ndarray] = None
        ):
        self.filter_indices = filter_indices
        self.values = values
        if nrows is not None:
            self.nrows = nrows
        else:
            self.nrows = len(filter_indices)

    def to_dense(self):
        pass

    def to_pd(self):
        pass

    @classmethod
    def from_pd_sparse(cls, indices: pd.Series, values: pd.Series):
        pass

    @classmethod
    def from_pd_dense(cls, values: pd.Series):
        pass

class VectorCat(Vector):

    def __init__(
            self,
            filter_indices: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
            values: Optional[np.ndarray] = None
        ):
        pass

    def to_dense(self):
        pass

    def to_pd(self):
        pass

    @classmethod
    def from_pd_sparse(cls, indices: pd.Series, values: pd.Series):
        pass

    @classmethod
    def from_pd_dense(cls, values: pd.Series):
        pass

class VectorNum(Vector):

    def __init__(
            self,
            filter_indices: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
            values: Optional[np.ndarray] = None
        ):
        pass

    def to_dense(self):
        pass

    def to_pd(self):
        pass

    @classmethod
    def from_pd_sparse(cls, indices: pd.Series, values: pd.Series):
        pass

    @classmethod
    def from_pd_dense(cls, values: pd.Series):
        pass

class Matrix:

    def __init__(
            self,
            vectors: List[Union["Matrix", Vector]]
        ):
        # Save list of filter_indices
        # Save list of values
        pass

    def to_dense(self):
        pass

    def to_pd(self):
        pass

    @classmethod
    def from_pd_sparse(cls, indices: pd.DataFrame, values: pd.DataFrame):
        pass

    @classmethod
    def from_pd_dense(cls, values: pd.DataFrame):
        pass

if __name__ == '__main__':
    # Vector Num: indices (or None), values (or None)
    # Vector Cat: dict(indices (or None))
    # Matrix: dict of vectors

    vc1 = VectorCat.from_dense()
    vc2 = VectorCat.from_sparse()
    vn1 = VectorNum.from_dense()
    vn2 = VectorNum.from_sparse()
    val = Matrix([vc1, vc2, vn1, vn2])

