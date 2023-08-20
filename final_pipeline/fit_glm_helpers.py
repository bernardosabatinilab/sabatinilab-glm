from typing import DefaultDict, List, TypeVar, Optional, Union, Set, Tuple, Dict, Any
import numpy as np
import pandas as pd
import scipy.sparse

class Vector():

    def __init__(
            self,
            name: str,
            nrows: int,
            values: Optional[np.ndarray] = None,
            indices: Optional[np.ndarray] = None,
            fill_value: Any = 0,
            filter_names: Set[str] = None,
        ):
        self.name = name
        self.filter_names = filter_names
        self.nrows = nrows
        self.fill_value = fill_value

        if indices is None and values is None:
            raise ValueError("At least one of indices and values must be specified")

        if values is None:
            self.values = None
        elif type(values) == np.ndarray:
            if self.fill_value in values:
                raise ValueError("values cannot contain fill_value")
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
    
    def intersect(self, indices: np.ndarray = None, filter_name: Set[str] = set()):
        assert type(filter_name) == set and all([type(x) == str for x in filter_name]), "filter_name must be a set of strings"
        filter_names = self.filter_names.union(filter_name)
        if np.all(self.indices == indices):
            return Vector(
                name=self.name,
                nrows=self.nrows,
                values=self.values,
                indices=self.indices,
                fill_value=self.fill_value,
                filter_names=filter_names
            )
        indices_prior = np.arange(self.nrows) if self.indices is None else self.indices
        kept_indices = np.intersect1d(indices_prior, indices)
        indices_in_kept = np.isin(indices_prior, kept_indices)
        values = self.values[indices_in_kept] if self.values is not None else None
        indices = indices_prior[indices_in_kept]
        return Vector(
            name=self.name,
            nrows=self.nrows,
            values=values,
            indices=indices,
            fill_value=self.fill_value,
            filter_names=filter_names
        )
    
    def __and__(self, other: "Vector"):
        if self.values is not None and other.values is not None:
            raise ValueError("At most, one vector can have values")
        if self.nrows != other.nrows:
            raise ValueError("Vectors must have the same nrows")
        if self.fill_value != other.fill_value:
            raise ValueError("Vectors must have the same fill_value")
        
        self_filter_names = self.filter_names if self.filter_names is None else set()
        other_filter_names = other.filter_names if other.filter_names is None else set()
        filter_names = self_filter_names.union(other_filter_names)

        if other.values is None:
            return self.intersect(other.indices, filter_names.add(other.name))
        else:
            return other.intersect(self.indices, filter_names.add(self.name))
        
    def __repr__(self) -> str:
        return f"Vector(name={self.name}, nrows={self.nrows}, values={self.values}, indices={self.indices}, fill_value={self.fill_value}, filter_names={self.filter_names})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __eq__(self, other: "Vector") -> bool:
        return self.name == other.name and self.nrows == other.nrows and self.fill_value == other.fill_value and np.all(self.values == other.values) and np.all(self.indices == other.indices) and self.filter_names == other.filter_names
    
    def __ne__(self, other: "Vector") -> bool:
        return not self.__eq__(other)
    
    def __len__(self) -> int:
        return self.nrows

    def shift(self, shift: int):
        return Vector(
            name=self.name,
            nrows=self.nrows,
            values=self.values,
            indices=self.indices + shift,
            fill_value=self.fill_value,
            filter_names=self.filter_names
        )


class VectorCategory(Vector):

    def __init__(
            self,
            name: str,
            nrows: int,
            values: np.ndarray,
            indices: Optional[np.ndarray] = None,
            fill_value: Any = np.nan,
        ):
        indices = np.arange(nrows) if indices is None else indices
        super(VectorCategory, self).__init__(name, nrows, values, indices, fill_value)
        categories_unique = np.unique(values)
        if len(categories_unique) < 2:
            raise ValueError("values for VectorCategory must have at least two unique values")
        self.categories = {
            cat: Vector(
                name=str(cat),
                nrows=nrows,
                values=None,
                indices=indices[values == cat],
                fill_value=fill_value
            ) for cat in categories_unique
        }
    
    def intersect(self, indices: np.ndarray = None, filter_name: Set[str] = set()):
        assert type(filter_name) == set and all([type(x) == str for x in filter_name]), "filter_name must be a set of strings"
        filter_names = self.filter_names.union(filter_name)
        if np.all(self.indices == indices):
            return VectorCategory(
                name=self.name,
                nrows=self.nrows,
                values=self.values,
                indices=self.indices,
                fill_value=self.fill_value,
                filter_names=filter_names
            )
        indices_prior = np.arange(self.nrows) if self.indices is None else self.indices
        kept_indices = np.intersect1d(indices_prior, indices)
        indices_in_kept = np.isin(indices_prior, kept_indices)
        values = self.values[indices_in_kept] if self.values is not None else None
        indices = indices_prior[indices_in_kept]
        return VectorCategory(
            name=self.name,
            nrows=self.nrows,
            values=values,
            indices=indices,
            fill_value=self.fill_value,
            filter_names=filter_names
        )

class Matrix:

    def __init__(
            self,
            vectors: List[Union["Matrix", Vector]]
        ):
        # Save list of filter_indices
        # Save list of values
        pass

if __name__ == '__main__':
    # Vector Num: indices (or None), values (or None)
    # Vector Cat: dict(indices (or None))
    # Matrix: dict of vectors

    vc1 = VectorCategory.from_dense()
    vc2 = VectorCategory.from_sparse()
    vn1 = VectorNum.from_dense()
    vn2 = VectorNum.from_sparse()
    val = Matrix([vc1, vc2, vn1, vn2])



        
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