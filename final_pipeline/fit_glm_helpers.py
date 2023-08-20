from typing import DefaultDict, List, TypeVar, Optional, Union, Set, Tuple, Dict, Any
import numpy as np
import pandas as pd
import scipy.sparse

class VectorSparse():
    """
    A sparse vector with a name and a number of rows. Can be intersected with another VectorSparse
    where at most one has values. If neither has values, the intersection is the intersection of
    the indices. If only one has indices, the intersection is the values at those indices.
    If both have indices, the intersection is the values at the intersection of the indices.

    Parameters
    ----------
    name: str
        The name of the VectorSparse
    nrows: int
        The number of rows in the VectorSparse
    values: np.ndarray
        The values of the VectorSparse. If None, the VectorSparse is sparse.
    indices: np.ndarray
        The indices of the VectorSparse. If None, the VectorSparse is dense.
    sparse_value: Any
        The value to use for the sparse entries of the VectorSparse
    filter_names: Set[str]
        The names of the vectors that have been intersected to create this VectorSparse

    Attributes
    ----------
    name: str
        The name of the VectorSparse
    nrows: int
        The number of rows in the VectorSparse
    values: np.ndarray
        The values of the VectorSparse. If None, the VectorSparse is sparse.
    indices: np.ndarray
        The indices of the VectorSparse. If None, the VectorSparse is dense.
    sparse_value: Any
        The value to use for the sparse entries of the VectorSparse
    filter_names: Set[str]
        The names of the vectors that have been intersected to create this VectorSparse

    Methods
    -------
    intersect(indices: np.ndarray = None, filter_name: Set[str] = set()) -> VectorSparse
        Intersects this VectorSparse with another VectorSparse. If both have values, raises an error.
        If neither have values, the intersection is the intersection of the indices. If only one has
        indices, the intersection is the values at those indices. If both have indices, the intersection
        is the values at the intersection of the indices.
    __and__(other: VectorSparse) -> VectorSparse
        Intersects this VectorSparse with another VectorSparse. If both have values, raises an error.
        If neither have values, the intersection is the intersection of the indices. If only one has
        indices, the intersection is the values at those indices. If both have indices, the intersection
        is the values at the intersection of the indices.
    __repr__() -> str
        Returns a string representation of the VectorSparse
    __str__() -> str
        Returns a string representation of the VectorSparse
    __eq__(other: VectorSparse) -> bool 
        Returns True if the vectors are equal, False otherwise
    __ne__(other: VectorSparse) -> bool
        Returns True if the vectors are not equal, False otherwise
    __len__() -> int
        Returns the number of rows in the VectorSparse
    shift(shift: int) -> VectorSparse
        Shifts the indices of the VectorSparse by the specified amount
    to_dense() -> np.ndarray
        Returns the dense representation of the VectorSparse
    to_pd() -> pd.Series
        Returns the dense representation of the VectorSparse as a pandas Series
    """

    def __init__(
            self,
            name: str,
            nrows: int,
            values: Optional[np.ndarray] = None,
            indices: Optional[np.ndarray] = None,
            sparse_value: Any = 0,
            filter_names: Set[str] = None,
        ):
        self.name = name
        self.filter_names = filter_names
        self.nrows = nrows
        self.sparse_value = sparse_value

        if indices is None and values is None:
            raise ValueError("At least one of indices and values must be specified")

        if values is None:
            self.values = None
        elif type(values) == np.ndarray:
            if self.sparse_value in values:
                raise ValueError("values cannot contain sparse_value")
            self.values = values
        else:
            raise ValueError("values must be a numpy array")
        
        if indices is None:
            self.indices = None
        elif type(indices) == np.ndarray:
            self.indices = indices
        else:
            raise ValueError("indices must be a numpy array")
        
        if self.indices is not None and self.values is not None:
            if len(self.indices) != len(self.values):
                raise ValueError("indices and values must be the same length")
    
    def intersect(self, indices: np.ndarray = None, filter_name: Set[str] = set()) -> "VectorSparse":
        """
        Intersects this VectorSparse with another VectorSparse. If both have values, raises an error.
        If neither have values, the intersection is the intersection of the indices. If only one has
        indices, the intersection is the values at those indices.

        Parameters
        ----------
        indices: np.ndarray
            The indices to intersect with
        filter_name: Set[str]
            The names of the vectors that have been intersected to create this VectorSparse

        Returns
        -------
        VectorSparse
            The intersection of this VectorSparse with the other VectorSparse
        """
        assert type(filter_name) == set and all([type(x) == str for x in filter_name]), "filter_name must be a set of strings"
        filter_names = self.filter_names if self.filter_names is not None else set()
        filter_names = filter_names.union(filter_name)
        if len(self.indices) == len(indices) and np.all(self.indices == indices):
            return self.__class__(
                name=self.name,
                nrows=self.nrows,
                values=self.values,
                indices=self.indices,
                sparse_value=self.sparse_value,
                filter_names=filter_names
            )
        indices_prior = np.arange(self.nrows) if self.indices is None else self.indices
        kept_indices = np.intersect1d(indices_prior, indices)
        indices_in_kept = np.isin(indices_prior, kept_indices)
        values = self.values[indices_in_kept] if self.values is not None else None
        indices = indices_prior[indices_in_kept]
        return self.__class__(
            name=self.name,
            nrows=self.nrows,
            values=values,
            indices=indices,
            sparse_value=self.sparse_value,
            filter_names=filter_names
        )
    
    def __and__(self, other: "VectorSparse") -> "VectorSparse":
        """
        Intersects this VectorSparse with another VectorSparse. If both have values, raises an error.
        If neither have values, the intersection is the intersection of the indices. If only one has
        indices, the intersection is the values at those indices.

        Parameters
        ----------
        other: VectorSparse
            The VectorSparse to intersect with
            
        Returns
        -------
        VectorSparse
            The intersection of this VectorSparse with the other VectorSparse
        """
        if self.values is not None and other.values is not None:
            raise ValueError("At most, one VectorSparse can have values")
        if self.nrows != other.nrows:
            raise ValueError("vectors must have the same nrows")
        
        self_filter_names = self.filter_names if self.filter_names is not None else set()
        other_filter_names = other.filter_names if other.filter_names is not None else set()
        filter_names = self_filter_names.union(other_filter_names)

        if other.values is None:
            return self.intersect(other.indices, filter_names.union({other.name}))
        else:
            return other.intersect(self.indices, filter_names.union({self.name}))
        
    def __repr__(self) -> str:
        """
        Returns a string representation of the VectorSparse

        Returns
        -------
        str
            A string representation of the VectorSparse
        """
        return f"VectorSparse(name={self.name}, nrows={self.nrows}, values={self.values}, indices={self.indices}, sparse_value={self.sparse_value}, filter_names={self.filter_names})"
    
    def __str__(self) -> str:
        """
        Returns a string representation of the VectorSparse

        Returns
        -------
        str
            A string representation of the VectorSparse
        """
        return self.__repr__()
    
    def __eq__(self, other: "VectorSparse") -> bool:
        """
        Returns True if the vectors are equal, False otherwise

        Parameters
        ----------
        other: VectorSparse
            The VectorSparse to compare to

        Returns
        -------
        bool
            True if the vectors are equal, False otherwise
        """
        return self.name == other.name and self.nrows == other.nrows and self.sparse_value == other.sparse_value and np.all(self.values == other.values) and np.all(self.indices == other.indices) and self.filter_names == other.filter_names
    
    def __ne__(self, other: "VectorSparse") -> bool:
        """
        Returns True if the vectors are not equal, False otherwise

        Parameters
        ----------
        other: VectorSparse
            The VectorSparse to compare to

        Returns
        -------
        bool
            True if the vectors are not equal, False otherwise
        """
        return not self.__eq__(other)
    
    def __len__(self) -> int:
        """
        Returns the number of rows in the VectorSparse

        Returns
        -------
        int
            The number of rows in the VectorSparse
        """
        return self.nrows

    def shift(self, shift: int) -> "VectorSparse":
        """
        Shifts the indices of the VectorSparse by the specified amount

        Parameters
        ----------
        shift: int
            The amount to shift the indices by
            
        Returns
        -------
        VectorSparse
            The VectorSparse with the shifted indices
        """
        return self.__class__(
            name=self.name,
            nrows=self.nrows,
            values=self.values,
            indices=self.indices + shift,
            sparse_value=self.sparse_value,
            filter_names=self.filter_names
        )
    
    def to_dense(self) -> np.ndarray:
        """
        Returns the dense representation of the VectorSparse

        Returns
        -------
        np.ndarray
            The dense representation of the VectorSparse

        Notes
        -----
        If the VectorSparse is sparse, the dense representation is a numpy array of length nrows
        with the sparse_value at the indices and 0 elsewhere. If the VectorSparse is dense, the
        dense representation is the values.
        """
        if self.indices is None:
            return self.values
        else:
            dense = np.full(self.nrows, self.sparse_value)
            indices = self.indices if self.indices is not None else np.arange(self.nrows)
            values = self.values if self.values is not None else np.ones(len(self.indices))
            dense[indices] = values
            return dense
    
    def to_pd(self) -> pd.Series:
        """
        Returns the dense representation of the VectorSparse as a pandas Series

        Returns
        -------
        pd.Series
            The dense representation of the VectorSparse as a pandas Series

        Notes
        -----
        If the VectorSparse is sparse, the dense representation is a pandas Series of length nrows
        with the sparse_value at the indices and 0 elsewhere. If the VectorSparse is dense, the
        dense representation is the values.

        The name of the pandas Series is the name of the VectorSparse.
        """
        return pd.Series(self.to_dense(), name=self.name)
    

class VectorSparse_Category(VectorSparse):
    """
    A sparse vector with a name and a number of rows. Can be intersected with another VectorSparse
    where at most one has values. If neither has values, the intersection is the intersection of
    the indices. If only one has indices, the intersection is the values at those indices.
    If both have indices, the intersection is the values at the intersection of the indices.

    Parameters
    ----------
    name: str
        The name of the VectorSparse
    nrows: int
        The number of rows in the VectorSparse
    values: np.ndarray
        The values of the VectorSparse. If None, the VectorSparse is sparse.
    indices: np.ndarray
        The indices of the VectorSparse. If None, the VectorSparse is dense.
    sparse_value: Any
        The value to use for the sparse entries of the VectorSparse
    filter_names: Set[str]
        The names of the vectors that have been intersected to create this VectorSparse

    Attributes
    ----------
    name: str
        The name of the VectorSparse
    nrows: int
        The number of rows in the VectorSparse
    values: np.ndarray
        The values of the VectorSparse. If None, the VectorSparse is sparse.
    indices: np.ndarray
        The indices of the VectorSparse. If None, the VectorSparse is dense.
    sparse_value: Any
        The value to use for the sparse entries of the VectorSparse
    filter_names: Set[str]
        The names of the vectors that have been intersected to create this VectorSparse

    Methods
    -------
    intersect(indices: np.ndarray = None, filter_name: Set[str] = set()) -> VectorSparse
        Intersects this VectorSparse with another VectorSparse. If both have values, raises an error.
        If neither have values, the intersection is the intersection of the indices. If only one has
        indices, the intersection is the values at those indices. If both have indices, the intersection
        is the values at the intersection of the indices.
    __and__(other: VectorSparse) -> VectorSparse
        Intersects this VectorSparse with another VectorSparse. If both have values, raises an error.
        If neither have values, the intersection is the intersection of the indices. If only one has
        indices, the intersection is the values at those indices. If both have indices, the intersection
        is the values at the intersection of the indices.
    __repr__() -> str
        Returns a string representation of the VectorSparse
    __str__() -> str
        Returns a string representation of the VectorSparse
    __eq__(other: VectorSparse) -> bool
        Returns True if the vectors are equal, False otherwise
    __ne__(other: VectorSparse) -> bool
        Returns True if the vectors are not equal, False otherwise
    __len__() -> int
        Returns the number of rows in the VectorSparse
    shift(shift: int) -> VectorSparse
        Shifts the indices of the VectorSparse by the specified amount
    to_dense() -> np.ndarray
        Returns the dense representation of the VectorSparse
    to_pd() -> pd.Series
        Returns the dense representation of the VectorSparse as a pandas Series

    Notes
    -----
    The values of a VectorSparse_Category must be hashable
    """
    def __init__(
            self,
            name: str,
            nrows: int,
            values: np.ndarray,
            indices: Optional[np.ndarray] = None,
            sparse_value: Any = 0,
            filter_names: Set[str] = None,
        ):
        indices = np.arange(nrows) if indices is None else indices
        super(VectorSparse_Category, self).__init__(name, nrows, values, indices, sparse_value)
        categories_unique = np.unique(values)
        if len(categories_unique) < 2:
            raise ValueError("values for VectorSparse_Category must have at least two unique values")
        self.categories = {
            cat: VectorSparse(
                name=str(cat),
                nrows=nrows,
                values=None,
                indices=indices[values == cat],
                sparse_value=sparse_value,
                filter_names=filter_names
            ) for cat in categories_unique
        }
    
    def to_dense(self) -> np.ndarray:
        """
        Returns the dense representation of the VectorSparse

        Returns
        -------
        np.ndarray
            The dense representation of the VectorSparse
        """
        return np.stack([self.categories[cat].to_dense() for cat in self.categories], axis=1)
    
    def to_pd(self) -> pd.DataFrame:
        """
        Returns the dense representation of the VectorSparse as a pandas DataFrame

        Returns
        -------
        pd.DataFrame
            The dense representation of the VectorSparse as a pandas DataFrame
        """
        return pd.DataFrame(
            self.to_dense(),
            columns=pd.MultiIndex.from_tuples(list(zip([self.name]*len(self.categories), self.categories.keys())))
        )
    

class Matrix:
    """
    A matrix of VectorSparse objects. Can be intersected with another Matrix. The intersection
    of two Matrices is the intersection of all the VectorSparse objects in the Matrices.

    Parameters
    ----------
    vectors: List[Union[Matrix, VectorSparse, VectorSparse_Category]]
        The vectors to include in the Matrix

    Attributes
    ----------
    vectors: Dict[Union[str, Tuple[str, Tuple[str]]], Union[VectorSparse, VectorSparse_Category]]
        The vectors in the Matrix

    Methods
    -------
    __and__(other: Union[Matrix, VectorSparse]) -> Matrix
        Intersects this Matrix with another Matrix or VectorSparse. The intersection of two Matrices
        is the intersection of all the VectorSparse objects in the Matrices.
    __getitem__(key: Any) -> Union[Matrix, VectorSparse, VectorSparse_Category]
        Returns the VectorSparse with the specified key
    intersect(indices: np.ndarray = None, filter_name: Set[str] = set()) -> Matrix
        Intersects this Matrix with another Matrix or VectorSparse. The intersection of two Matrices
        is the intersection of all the VectorSparse objects in the Matrices.
    to_dense() -> np.ndarray
        Returns the dense representation of the Matrix
    to_pd() -> pd.DataFrame
        Returns the dense representation of the Matrix as a pandas DataFrame

    Notes
    -----
    The names of the VectorSparse objects in a Matrix must be unique
    """
    def __init__(
            self,
            vectors: List[Union["Matrix", VectorSparse, VectorSparse_Category]]
        ):
        if type(vectors) != list or not all([type(x) in [VectorSparse_Category, VectorSparse, Matrix] for x in vectors]):
            raise ValueError("vectors must be a list of vectors or Matrices")
        
        self.vectors = {}
        for vector in vectors:
            if vector.name in self.vectors:
                raise ValueError(f"vectors must have unique names, {vector.name} is duplicated")
            if vector.filter_names is None or len(vector.filter_names) == 0:
                self.vectors[vector.name] = vector
            else:
                self.vectors[(vector.name, tuple(sorted(set(vector.filter_names))))] = vector

    def __repr__(self) -> str:
        """
        Returns a string representation of the Matrix

        Returns
        -------
        str
            A string representation of the Matrix
        """
        return f"Matrix(vectors={self.vectors})"
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Matrix

        Returns
        -------
        str
            A string representation of the Matrix
        """
        return self.__repr__()
    
    def __and__(self, other: Union["Matrix", VectorSparse]) -> "Matrix":
        """
        Intersects this Matrix with another Matrix or VectorSparse. The intersection of two Matrices
        is the intersection of all the VectorSparse objects in the Matrices.

        Parameters
        ----------
        other: Union[Matrix, VectorSparse]
            The Matrix or VectorSparse to intersect with

        Returns
        -------
        Matrix
            The intersection of this Matrix with the other Matrix or VectorSparse
        """
        if type(other) == VectorSparse or type(other) == VectorSparse_Category:
            return Matrix([x & other for x in self.vectors])
        elif type(other) == Matrix:
            return Matrix([x & y for x in self.vectors for y in other.vectors])
        else:
            raise ValueError("other must be a VectorSparse or Matrix")
    
    def __getitem__(self, key: Any) -> Union["Matrix", VectorSparse, VectorSparse_Category]:
        """
        Returns the VectorSparse with the specified key

        Parameters
        ----------
        key: Any
            The key of the VectorSparse to return

        Returns
        -------
        Union[Matrix, VectorSparse, VectorSparse_Category]
            The VectorSparse with the specified key

        Notes
        -----
        If the key is a string, returns the VectorSparse with that name. If the key is a tuple of
        length 2, returns the VectorSparse with the first element of the tuple as the name and the
        second element of the tuple as the filter_names. If the key is a tuple of length 1, returns
        the VectorSparse with the first element of the tuple as the name and no filter_names.
        """
        return self.vectors[key]

    def intersect(self, indices: np.ndarray = None, filter_name: Set[str] = set()) -> "Matrix":
        """
        Intersects this Matrix with another Matrix or VectorSparse. The intersection of two Matrices
        is the intersection of all the VectorSparse objects in the Matrices.

        Parameters
        ----------
        indices: np.ndarray
            The indices to intersect with
        filter_name: Set[str]
            The names of the vectors that have been intersected to create this Matrix

        Returns
        -------
        Matrix
            The intersection of this Matrix with the other Matrix or VectorSparse
        """
        return self.__class__([x.intersect(indices, filter_name) for x in self.vectors])
    
    def to_dense(self) -> np.ndarray:
        """
        Returns the dense representation of the Matrix
        
        Returns
        -------
        np.ndarray
            The dense representation of the Matrix

        Notes
        -----
        The dense representation of the Matrix is a numpy array of shape (nrows, ncols) where
        nrows is the number of rows in the Matrix and ncols is the number of VectorSparse objects
        in the Matrix. The columns of the numpy array are the dense representations of the
        VectorSparse objects in the Matrix.

        If a VectorSparse object is sparse, the dense representation is a numpy array of length nrows
        with the sparse_value at the indices and 0 elsewhere. If a VectorSparse object is dense, the
        dense representation is the values.

        If a VectorSparse_Category object is sparse, the dense representation is a numpy array of shape
        (nrows, ncategories) where nrows is the number of rows in the Matrix and ncategories is the
        number of categories in the VectorSparse_Category object. The columns of the numpy array are
        the dense representations of the categories in the VectorSparse_Category object. If a category
        is sparse, the dense representation is a numpy array of length nrows with the sparse_value at
        the indices and 0 elsewhere. If a category is dense, the dense representation is the values.
        """
        lst_dense = []
        for x in self.vectors.values():
            dense = x.to_dense()
            if len(dense.shape) == 1:
                dense = dense.reshape(-1, 1)
            lst_dense.append(dense)
        return np.concatenate(lst_dense, axis=1)
    
    def to_pd(self) -> pd.DataFrame:
        """
        Returns the dense representation of the Matrix as a pandas DataFrame

        Returns
        -------
        pd.DataFrame
            The dense representation of the Matrix as a pandas DataFrame
        """
        return pd.concat([x.to_pd() for x in self.vectors.values()], axis=1)

if __name__ == '__main__':
    # VectorSparse Num: indices (or None), values (or None)
    # VectorSparse Cat: dict(indices (or None))
    # Matrix: dict of vectors

    # vc1 = VectorSparse_Category.from_dense()
    # vc2 = VectorSparse_Category.from_sparse()
    # vn1 = VectorSparseNum.from_dense()
    # vn2 = VectorSparseNum.from_sparse()
    # val = Matrix([vc1, vc2, vn1, vn2])

    indices_a = np.array([0, 1, 2, 3, 4, 5])
    values_a = np.array([1, 2, 3, 4, 5, 6])
    vec_a = VectorSparse(name="a", nrows=12, values=values_a, indices=indices_a)
    indices_b = np.array([1, 2, 4, 5, 10])
    vec_b = VectorSparse(name="b", nrows=12, values=None, indices=indices_b)
    vec_c = vec_a & vec_b
    print(vec_a)
    print(vec_a.to_dense())
    print(vec_a.to_pd())
    print(vec_b)
    print(vec_b.to_dense())
    print(vec_b.to_pd())
    print(vec_c)
    print(vec_c.to_dense())
    print(vec_c.to_pd())

    indices_a = np.array([0, 1, 2, 3, 4, 5])
    values_a = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    vec_a = VectorSparse_Category(name="cat_col", nrows=11, values=values_a, indices=indices_a)
    indices_b = np.array([1, 2, 4, 5, 10])
    vec_b = VectorSparse(name="b", nrows=11, values=None, indices=indices_b)
    vec_c = vec_a & vec_b
    print(vec_a)
    print(vec_a.to_dense())
    print(vec_a.to_pd())
    print(vec_b)
    print(vec_b.to_dense())
    print(vec_b.to_pd())
    print(vec_c)
    print(vec_c.to_dense())
    print(vec_c.to_pd())
    values_d = np.array(['c', 'c', 'c', 'd', 'd', 'd'])
    vec_d = VectorSparse_Category(name="cat_col2", nrows=11, values=values_d, indices=indices_a).shift(2)

    print(pd.concat([vec_a.to_pd(), vec_d.to_pd()], axis=1))

    matrix = Matrix([vec_a, vec_d, vec_b])
    print(matrix.to_dense())
    print(matrix.to_pd())

    print()
        
    # def filter(self, filter_indices: np.ndarray):
    #     if self.indices is None:
    #         self.indices = filter_indices
    #     else:
    #         self.indices = np.intersect1d(self.indices, filter_indices)
    #     elif type(filter_indices) in [list, np.ndarray]:
    #         lst_filter_indices = [filter_indices] if type(filter_indices) == np.ndarray else filter_indices
    #         if not all([type(x) == np.ndarray for x in filter_indices]):
    #             raise ValueError("filter_indices must be a numpy array or a list of numpy arrays")
    #         lst_filter_indices = filter_indices
    #         indices = np.arange(self.nrows)
    #         for filter_index in lst_filter_indices:
    #             indices = np.intersect1d(indices, filter_index)
    #         self.indices = indices
    #     else:
    #         raise ValueError("filter_indices must be a numpy array or a list of numpy arrays")

    #     if values is None:
    #         self.values = None
    #     elif type(values) == np.ndarray:
    #         if self.indices is not None and len(values) != len(self.indices):
    #             raise ValueError("values must be the same length as filter_indices if both specified")
                
    #         self.values = values



    # def to_dense(self):
    #     pass

    # def to_pd(self):
    #     pass

    # @classmethod
    # def from_pd_sparse(cls, indices: pd.Series, values: pd.Series):
    #     pass

    # @classmethod
    # def from_pd_dense(cls, values: pd.Series):
    #     pass