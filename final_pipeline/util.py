from typing import DefaultDict, List, TypeVar, Optional, Union, Set, Tuple, Dict, Any
import numpy as np
import pandas as pd
import scipy.sparse
from collections.abc import Iterable

class VectorSparse:
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
            shifted_amount: int = 0
        ):
        self.name = name
        self.filter_names = filter_names
        self.nrows = nrows
        self.sparse_value = sparse_value
        self.shifted_amount = shifted_amount

        if indices is None and values is None:
            raise ValueError("At least one of indices and values must be specified")

        if values is None:
            self.values = None
        elif isinstance(values, np.ndarray):
            if self.sparse_value in values:
                raise ValueError("values cannot contain sparse_value")
            self.values = values
        else:
            raise ValueError("values must be a numpy array")
        
        if indices is None:
            self.indices = None
        elif isinstance(indices, np.ndarray):
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
        if not isinstance(filter_name, set):
            assert False, "filter_name must be a set of strings or a set of strings and tuples of strings"
        else:
            for entry in filter_name:
                if isinstance(entry, str):
                    pass
                elif isinstance(entry, tuple):
                    for subentry in entry:
                        assert isinstance(subentry, str), "filter_name must be a set of strings or a set of strings and tuples of strings"
                else:
                    assert False, "filter_name must be a set of strings or a set of strings and tuples of strings"
        
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
            return self.intersect(other.indices, filter_names)
        else:
            return other.intersect(self.indices, filter_names)
    
    def __invert__(self) -> "VectorSparse":
        """
        Returns a VectorSparse with the values negated

        Returns
        -------
        VectorSparse
            A VectorSparse with the values negated
        """
        assert self.values is None, "VectorSparse must not have values in order to negate"
        inverted_indices = np.setdiff1d(np.arange(self.nrows), self.indices)
        return self.__class__(
            name=self.name,
            nrows=self.nrows,
            values=None,
            indices=inverted_indices,
            sparse_value=-self.sparse_value,
            filter_names=set({str(self.filter_names.union({'~'}))}), #set([f'~{filter_name}' for filter_name in list(self.filter_names)])
        )
    
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

    def shift(self, shift: Union[int, List[int]]) -> Union["VectorSparse", "Matrix"]:
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
        if isinstance(shift, int):
            indices_init = self.indices + shift
            values_init = self.values if self.values is not None else np.ones(len(indices_init))
            
            if indices_init is not None:
                inx_in_bounds = (indices_init >= 0)&(indices_init < self.nrows)
                indices_init = indices_init[inx_in_bounds]
                if values_init is not None:
                    values_init = values_init[inx_in_bounds]

            if np.isnan(self.sparse_value) or shift == 0:
                values = values_init if values_init is not None else None
                indices = indices_init if indices_init is not None else None
            elif shift > 0:
                values = np.concatenate([np.full(shift, np.nan), values_init]) if values_init is not None else None
                indices = np.concatenate([np.arange(shift), indices_init]) if indices_init is not None else None
            else:
                values = np.concatenate([values_init, np.full(np.abs(shift), np.nan)]) if values_init is not None else None
                indices = np.concatenate([indices_init, self.nrows + np.arange(shift, 0)]) if indices_init is not None else None
            
            vector_shifted = self.__class__(
                name=self.name,
                nrows=self.nrows,
                values=values,
                indices=indices,
                sparse_value=self.sparse_value,
                filter_names=self.filter_names,
            )
            vector_shifted.shifted_amount = shift
            return vector_shifted
        elif isinstance(shift, list) and all([isinstance(x, int) for x in shift]):
            return Matrix([self.shift(x) for x in shift])
        else:
            raise ValueError("shift must be an int or a list of ints")
    
    def one_hots_to_inx(self) -> "VectorSparse":
        """
        Converts a one-hot encoded VectorSparse to an index VectorSparse

        Returns
        -------
        VectorSparse
            The index VectorSparse
        """
        assert len(np.unique(self.values)) == 1, "VectorSparse must have one unique value"
        assert np.unique(self.values)[0] == 1, "Only value in VectorSparse must be 1"
        assert self.sparse_value == 0, "sparse_value must be 0"
        assert self.indices is not None, "VectorSparse must have indices"
        assert self.values is not None, "VectorSparse must have values"

        name = ''
        name_set = {self.name}
        filter_names = self.filter_names.union(name_set) if self.filter_names is not None else name_set
        
        return self.__class__(
            name=name,
            nrows=self.nrows,
            values=None,
            indices=self.indices,
            sparse_value=self.sparse_value,
            filter_names=filter_names
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
            indices = self.indices if self.indices is not None else np.arange(self.nrows)
            values = self.values if self.values is not None else np.ones(len(self.indices))
            
            if object in [values.dtype, type(self.sparse_value)]:
                dtype = object
            elif float in [values.dtype, type(self.sparse_value)]:
                dtype = float
            elif int in [values.dtype, type(self.sparse_value)]:
                dtype = int
            else:
                dtype = values.dtype
            
            dense = np.full(self.nrows, self.sparse_value, dtype=dtype)
            dense[indices] = values
            return dense
    
    def to_pd(self, keep_shifted_amount_column: bool = False) -> pd.Series:
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
        base_name_set = {self.name} if self.name is not None else set()
        base_name_set = base_name_set.union(set(self.filter_names)) if self.filter_names is not None else base_name_set
        if len(base_name_set) == 0:
            raise ValueError("VectorSparse must have a name")
        # elif len(base_name_set) == 1:
        #     name = (base_name_set.pop(), self.shifted_amount) if keep_shifted_amount_column else base_name_set.pop()
        else:
            base_filter_name_set = set(self.filter_names).copy() if self.filter_names is not None else set()
            if self.filter_names is None or len(self.filter_names) == 0:
                filter_names = ''
            elif len(self.filter_names) == 1:
                filter_names = base_filter_name_set.pop()
            else:
                filter_names = tuple(sorted(base_filter_name_set))
            name = (self.name, filter_names, self.shifted_amount) if keep_shifted_amount_column else (self.name, filter_names)
        return pd.Series(self.to_dense(), name=name)

    def set_sparse_value(self, sparse_value: Any) -> "VectorSparse":
        """
        Returns self with a new sparse_value after setting the value inplace

        Parameters
        ----------
        sparse_value: Any
            The new sparse_value of the VectorSparse
        
        Returns
        -------
        self
        """
        self.sparse_value = sparse_value
        return self

    def rename(self, name: Union[str, Tuple[str, Tuple[str]]]) -> "VectorSparse":
        """
        Returns a new VectorSparse with a renamed value

        Parameters
        ----------
        name: Union[str, Tuple[str, Tuple[str]]]
            The new name of the VectorSparse
        
        Returns
        -------
        VectorSparse
            The new VectorSparse with the renamed value
        """
        return self.__class__(
            name=name,
            nrows=self.nrows,
            values=self.values,
            indices=self.indices,
            sparse_value=self.sparse_value,
            filter_names=self.filter_names
        )

    @classmethod
    def from_pd_dense(cls, series: pd.Series, sparse_value: Any = 0):
        """
        Creates a VectorSparse from a pandas Series

        Parameters
        ----------
        values: pd.Series
            The pandas Series to create the VectorSparse from

        Returns
        -------
        VectorSparse
            The VectorSparse created from the pandas Series

        Notes
        -----
        The name of the VectorSparse is the name of the pandas Series.
        """
        nrows = len(series)
        indices = np.arange(nrows)[series.values != sparse_value]
        values = series.values[indices]
        return cls(
            name=series.name,
            nrows=nrows,
            values=values,
            indices=indices,
            sparse_value=sparse_value,
            filter_names=None
        )
    
    @classmethod
    def from_pd_sparse(cls, nrows: int, indices: pd.Series, values: pd.Series, sparse_value: Any = 0):
        """
        Creates a VectorSparse from a pandas Series

        Parameters
        ----------
        values: pd.Series
            The pandas Series to create the VectorSparse from

        Returns
        -------
        VectorSparse
            The VectorSparse created from the pandas Series

        Notes
        -----
        The name of the VectorSparse is the name of the pandas Series.
        """
        name = values.name if values is not None else ''
        filter_name = {indices.name} if indices is not None else set()
        base_values = values.values if values is not None else None
        base_indices = indices.values if indices is not None else None
        return cls(
            name=name,
            nrows=nrows,
            values=base_values,
            indices=base_indices,
            sparse_value=sparse_value,
            filter_names=filter_name
        )

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
            shifted_amount: int = 0
        ):
        indices = np.arange(nrows) if indices is None else indices
        super(VectorSparse_Category, self).__init__(name, nrows, values, indices, sparse_value, filter_names, shifted_amount)
        categories_unique = np.unique(values)
        if len(categories_unique) < 2:
            raise ValueError("values for VectorSparse_Category must have at least two unique values")
        self.categories = {
            cat: VectorSparse(
                name=str(cat),
                nrows=nrows,
                values=None,
                indices=indices[np.array([hash(value) for value in values]) == hash(cat)],
                sparse_value=sparse_value,
                filter_names=filter_names
            ) for cat in categories_unique
        }
    
    def __and__(self, other: Union[VectorSparse, "VectorSparse_Category"]) -> "VectorSparse_Category":
        """
        Intersects this VectorSparse_Category with another VectorSparse or VectorSparse_Category.
        Calls the super __and__ if other is not VectorSparse_Category. Otherwise, creates a new
        VectorSparse_Category with indices as the intersection of the indices of the two and the
        values as a tuple of the values of the two.

        Parameters
        ----------
        other: Union[VectorSparse, VectorSparse_Category]
            The VectorSparse or VectorSparse_Category to intersect with

        Returns
        -------
        VectorSparse
            The intersection of this VectorSparse_Category with the other VectorSparse or VectorSparse_Category
        """
        if not isinstance(other, VectorSparse_Category):
            return super(VectorSparse_Category, self).__and__(other)
        if self.values is None or other.values is None:
            raise ValueError("Both VectorSparse_Categories must have values")
        if self.nrows != other.nrows:
            raise ValueError("vectors must have the same nrows")
        if self.sparse_value != other.sparse_value:
            raise ValueError("vectors must have the same sparse_value")
        
        self_filter_names = self.filter_names if self.filter_names is not None else set()
        other_filter_names = other.filter_names if other.filter_names is not None else set()
        filter_names = self_filter_names.union(other_filter_names)

        value_names = (self.name, other.name)

        kept_indices = np.intersect1d(self.indices, other.indices)
        self_indices_kept = np.isin(self.indices, kept_indices)
        other_indices_kept = np.isin(other.indices, kept_indices)

        self_indices = self.indices[self_indices_kept] if self.indices is not None else None
        other_indices = other.indices[other_indices_kept] if other.indices is not None else None
        assert np.all(self_indices == other_indices), "Kept indices must be the same for both VectorSparse_Categories"
        indices = self_indices

        self_values = self.values[self_indices_kept] if self.values is not None else None
        other_values = other.values[other_indices_kept] if other.values is not None else None

        # Source: https://stackoverflow.com/questions/47389447/how-convert-a-list-of-tuples-to-a-numpy-array-of-tuples
        values_setup = [tuple(entry) for entry in np.stack([self_values, other_values], axis=1)]
        values = np.empty(len(values_setup), dtype=object)
        values[:] = values_setup

        return self.__class__(
            name=value_names,
            nrows=self.nrows,
            values=values,
            indices=indices,
            sparse_value=self.sparse_value,
            filter_names=filter_names
        )
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the VectorSparse

        Returns
        -------
        str
            A string representation of the VectorSparse
        """
        return f"VectorSparse_Category(name={self.name}, nrows={self.nrows}, values={self.values}, indices={self.indices}, sparse_value={self.sparse_value}, filter_names={self.filter_names})"
    
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
        
        # return pd.DataFrame(
        #     self.to_dense(),
        #     columns=pd.MultiIndex.from_tuples(list(zip([self.name]*len(self.categories), self.categories.keys())))
        # )

        return self.to_matrix().to_pd()
    
    def set_sparse_value(self, sparse_value: Any) -> "VectorSparse":
        """
        Returns a new VectorSparse with a new sparse_value

        Parameters
        ----------
        sparse_value: Any
            The new sparse_value of the VectorSparse
        
        Returns
        -------
        VectorSparse
            The new VectorSparse with the new sparse_value
        """
        super(VectorSparse_Category, self).set_sparse_value(sparse_value)
        for cat, vector in self.categories.items():
            vector.set_sparse_value(sparse_value)

    def to_matrix(self) -> "Matrix":
        """
        Returns the dense representation of the VectorSparse as a Matrix

        Returns
        -------
        Matrix
            The dense representation of the VectorSparse as a Matrix
        """
        return Matrix([self.categories[cat].rename((self.name, cat)) for cat in self.categories])

    @classmethod
    def from_pd_dense(cls, series: pd.Series, sparse_value: Any = 0):
        """
        Creates a VectorSparse from a pandas Series

        Parameters
        ----------
        series: pd.Series
            The pandas Series to create the VectorSparse from

        Returns
        -------
        VectorSparse
            The VectorSparse created from the pandas Series

        Notes
        -----
        The name of the VectorSparse is the name of the pandas Series.
        """
        nrows = len(series)
        values = series.values
        return cls(
            name=series.name,
            nrows=nrows,
            values=values,
            indices=None,
            sparse_value=sparse_value,
            filter_names=None
        )
    
    @classmethod
    def from_pd_sparse(cls, nrows: int, indices: Optional[pd.Series] = None, values: Optional[pd.Series] = None, sparse_value: Any = 0):
        """
        Creates a VectorSparse from a pandas Series

        Parameters
        ----------
        values: pd.Series
            The pandas Series to create the VectorSparse from

        Returns
        -------
        VectorSparse
            The VectorSparse created from the pandas Series

        Notes
        -----
        The name of the VectorSparse is the name of the pandas Series.
        """
        return cls(
            name=values.name,
            nrows=nrows,
            values=values.values,
            indices=indices.values,
            sparse_value=sparse_value,
            filter_names={indices.name}
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
            components: List[Union["Matrix", VectorSparse, VectorSparse_Category]]
        ):
        if not isinstance(components, list) or not all([type(x) in [VectorSparse_Category, VectorSparse, Matrix] for x in components]):
            raise ValueError("vectors must be a list of vectors or Matrices")
        
        self.vectors = {}
        for component in components:
            if isinstance(component, VectorSparse_Category):
                component = component.to_matrix()
            if isinstance(component, Matrix):
                component = component.vectors
                for vector_key, vector_val in component.items():
                    key = (vector_key, vector_val.shifted_amount) if vector_val.shifted_amount else vector_key
                    if key in self.vectors:
                        raise ValueError(f"vectors must have unique names, {key} is duplicated")
                    self.vectors[key] = vector_val
            else:
                if component.filter_names is None or len(component.filter_names) == 0:
                    key = (component.name, component.shifted_amount) if component.shifted_amount else component.name
                    if key in self.vectors:
                        raise ValueError(f"vectors must have unique names, {key} is duplicated")
                    self.vectors[key] = component
                else:
                    key = (component.name, tuple(sorted(set(component.filter_names))), component.shifted_amount) if component.shifted_amount else (component.name, tuple(sorted(set(component.filter_names))))
                    if key in self.vectors:
                        raise ValueError(f"vectors must have unique names, {key} is duplicated")
                    self.vectors[key] = component

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
        if isinstance(other, VectorSparse) or isinstance(other, VectorSparse_Category):
            return Matrix([x & other for x in self.vectors])
        elif isinstance(other, Matrix):
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
        if isinstance(key, list):
            return Matrix([self[x] for x in key])
        else:
            key = (key,) if isinstance(key, str) else key
            vectors = {key_vector: vector for key_vector, vector in self.vectors.items()}
            for isubkey, subkey in enumerate(key):
                if isinstance(subkey, tuple) and len(subkey) == 1:
                    subkey = subkey[0]
                vectors = {vector_key: vector for vector_key, vector in vectors.items() if subkey is None or subkey == vector_key or (subkey in vector_key[isubkey] and len(subkey) > 0) or subkey == vector_key[isubkey]}
            if len(vectors) == 1:
                return list(vectors.values())[0]
            elif len(vectors) > 1:
                return Matrix([vector for vector in vectors.values()])
    
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
    
    def shift(self, shift: Union[int, List[int]]) -> Union["VectorSparse", "Matrix"]:
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
        if isinstance(shift, int):
            return Matrix([component.shift(shift) for component in self.vectors.values()])
        elif type(shift) in [list, range] and all([isinstance(x, int) for x in shift]):
            return Matrix([self.shift(x) for x in shift])
        else:
            raise ValueError("shift must be an int or a list of ints")
    
    def one_hots_to_inx(self) -> "Matrix":
        """
        Converts a one-hot encoded VectorSparse to an index VectorSparse

        Returns
        -------
        VectorSparse
            The index VectorSparse
        """
        return Matrix([x.one_hots_to_inx() for x in self.vectors.values()])
    
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
        keep_shifted_amount_column = len(set([x.shifted_amount for x in self.vectors.values()])) > 1
        return pd.concat([x.to_pd(keep_shifted_amount_column=keep_shifted_amount_column) for x in self.vectors.values()], axis=1)
    
    def set_sparse_value(self, sparse_value: Any) -> "VectorSparse":
        """
        Returns a new VectorSparse with a new sparse_value

        Parameters
        ----------
        sparse_value: Any
            The new sparse_value of the VectorSparse
        
        Returns
        -------
        VectorSparse
            The new VectorSparse with the new sparse_value
        """
        [vector.set_sparse_value(sparse_value) for key, vector in self.vectors.items()]
        return self
    
    @classmethod
    def from_pd_dense_num(cls, dataframe: Optional[pd.DataFrame] = None, sparse_value: Any = 0):
        """
        Creates a Matrix from a pandas DataFrame

        Parameters
        ----------
        dataframe: pd.DataFrame
            The pandas DataFrame to create the Matrix from

        Returns
        -------
        Matrix
            The Matrix created from the pandas DataFrame

        Notes
        -----
        The names of the VectorSparse objects in the Matrix are the column names of the pandas DataFrame.
        """
        return cls([VectorSparse.from_pd_dense(dataframe[col], sparse_value) for col in dataframe.columns])
    
    @classmethod
    def from_pd_dense_cat(cls, dataframe: Optional[pd.DataFrame] = None, sparse_value: Any = 0):
        """
        Creates a Matrix from a pandas DataFrame

        Parameters
        ----------
        dataframe: pd.DataFrame
            The pandas DataFrame to create the Matrix from

        Returns
        -------
        Matrix
            The Matrix created from the pandas DataFrame

        Notes
        -----
        The names of the VectorSparse objects in the Matrix are the column names of the pandas DataFrame.
        """
        return cls([VectorSparse_Category.from_pd_dense(dataframe[col], sparse_value) for col in dataframe.columns])
    
    @classmethod
    def from_pd_sparse_num(cls, nrows: int, indices: Optional[pd.DataFrame] = None, values: Optional[pd.DataFrame] = None, sparse_value: Any = 0):
        """
        Creates a Matrix from a pandas DataFrame

        Parameters
        ----------
        dataframe: pd.DataFrame
            The pandas DataFrame to create the Matrix from

        Returns
        -------
        Matrix
            The Matrix created from the pandas DataFrame

        Notes
        -----
        The names of the VectorSparse objects in the Matrix are the column names of the pandas DataFrame.
        """
        if indices is None and values is None:
            raise ValueError("indices and values cannot both be None")
        elif indices is None:
            return cls([VectorSparse.from_pd_sparse(nrows, None, values[val_col], sparse_value) for val_col in values.columns])
        elif values is None:
            return cls([VectorSparse.from_pd_sparse(nrows, indices[index_col], None, sparse_value) for index_col in indices.columns])
        else:
            return cls([VectorSparse.from_pd_sparse(nrows, indices[index_col], values[val_col], sparse_value) for index_col in indices.columns for val_col in values.columns])
    
    @classmethod
    def from_pd_sparse_cat(cls, nrows: int, indices: Optional[pd.DataFrame] = None, values: Optional[pd.DataFrame] = None, sparse_value: Any = 0):
        """
        Creates a Matrix from a pandas DataFrame

        Parameters
        ----------
        dataframe: pd.DataFrame
            The pandas DataFrame to create the Matrix from

        Returns
        -------
        Matrix
            The Matrix created from the pandas DataFrame

        Notes
        -----
        The names of the VectorSparse objects in the Matrix are the column names of the pandas DataFrame.
        """
        return cls([VectorSparse_Category.from_pd_sparse(nrows, indices[index_col], values[val_col], sparse_value) for index_col in indices.columns for val_col in values.columns])
    
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
    
    print(Matrix([vec_a, vec_d]).to_pd())
    print((vec_a & vec_d).to_pd())
    