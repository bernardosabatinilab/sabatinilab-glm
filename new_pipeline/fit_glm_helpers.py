from typing import DefaultDict, List, TypeVar, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import scipy.sparse


# Needed: Values and/or indices, but at least one. Specification of values as categorical or numerical. nrows. fill_values.
class Vector():
    def __init__(
            self,
            name: Optional[str]=None,
            values: Optional[np.ndarray]=None,
            indices: Optional[np.ndarray]=None,
            nrows: Optional[int]=None,
            # fill_values: Optional[float]=0, # Not yet implemented: Non-zero fill-values
            dtype: Optional[type]=None, # Use 'categorical' for categorical values # Non-categorical values are numerical # Other types not yet implemented
        ):

        assert (values is not None) or (indices is not None), 'At least one of values or indices must be specified.'

        self.fill_values = 0 # Not yet implemented: Non-zero fill-values
        self.name = (name,)
        self.indices = indices
        indices_arangeFilled = indices if indices is not None else np.arange(len(values))
        self.dtype = dtype if dtype is not None else 'numerical'
        self.nrows = nrows if nrows is not None else np.max(indices_arangeFilled) + 1

        if self.dtype != 'categorical':
            self.values = values
            self.ncols = 1
            self.subvectors = None
        else:
            self.values = None
            categories_unique = np.unique(values)
            self.ncols = len(categories_unique)
            self.subvectors = {}
            for category in categories_unique:
                subvector = Vector(
                    name=(name, category),
                    indices=indices_arangeFilled[values == category],
                    nrows=nrows,
                    dtype='numerical',
                )
                self.subvectors[(name, category)] = subvector

    def __repr__(self):
        if self.dtype == 'categorical':
            return f'Vector(dtype={self.dtype}, nrows={self.nrows}, submatrices={self.subvectors})'
        else:
            return f'Vector(dtype={self.dtype}, nrows={self.nrows}, values={self.values}, indices={self.indices})'
    
    def __str__(self):
        return self.__repr__()

class Matrix():
    def __init__(
            self,
            *subcomponents: List[Union[Vector, 'Matrix']],
    ):
        assert len(set([component.nrows for component in subcomponents])) == 1, 'All vectors in a matrix must have the same number of nrows.'

        self.ncols = sum([subcomponent.ncols for subcomponent in subcomponents])
        self.nrows = subcomponents[0].nrows
        self.subcomponents = {}
        for subcomponent in subcomponents:
            if isinstance(subcomponent, Vector):
                self.subcomponents[subcomponent.name] = subcomponent
            elif isinstance(subcomponent, Matrix):
                self.subcomponents = self.subcomponents | subcomponent.subcomponents
            else:
                raise TypeError(f'Unexpected type: {type(subcomponent)}')

    def __repr__(self):
        return f'Matrix(nrows={self.nrows}, ncols={self.ncols}, subcomponents={self.subcomponents})'
    
    def __str__(self):
        return self.__repr__()
    
    def __getitem__(self, key):
        assert not isinstance(key, slice), 'Slicing not allowed for Matrices.'

        lst_sliced_matrix_setup = []
        # If the get item key type is a list
        if isinstance(key, list):
            return Matrix(*[self[single_key] for single_key in key])
        
        # If the key type is a tuple return an intersection of the subcomponents' indices where only one subcomponent is allowed to have a non-none value
        elif isinstance(key, tuple):
            subcomponents_queried = [self[single_key] for single_key in key]
            # Assert values is not none for only one subcomponent
            assert sum([subcomponent.values is not None for subcomponent in subcomponents_queried]) != 1, 'A maximum of one subcomponent can have a non-none values.'
            
            # Create a list of all subcomponent indices
            lst_indices = [subcomponent.indices if subcomponent.indices is not None else np.arange(subcomponent.nrows) for subcomponent in subcomponents_queried]

            # Create a variable to represent the values of the single (or zero) subcomponent that has non-none values
            lst_values = [subcomponent.values for subcomponent in subcomponents_queried if subcomponent.values is not None]

            # Filter all indices and values to only include those that are in the intersection of the subcomponent indices
            for i, indices in enumerate(lst_indices):
                indices_intersected = indices if i == 0 else np.intersect1d(indices_intersected, indices)

            return Matrix(*lst_sliced_matrix_setup)

        # Otherwise, return the subcomponent corresponding to the key
        return self.subcomponents[key]

    



# Dense matrix constructed from just values (DV) gets associated with arange implied indices


# Sparse matrix constructed from just indices (SI) gets ones associated with specified indices and non-specified indices get zeros otherwise within shape of rows


# Sparse matrix constructed from both indices and values (SIV) gets values associated with indices and non-specified indices get zeros otherwise within shape of rows


# Dense categorical values (DC) are subsegmented into sparse matrices with each segmenttion of categories being indicated by the unique values of the categorical variable 


# Sparse categorical values (SC) are subsegmented into sparse matrices with each segmenttion of categories being indicated by the unique values of the categorical variable (except value 0 being unassociated)



class SparseMatrix():
    """
    A class for storing sparse matrices.

    JZ 2023
    """
    def __init__(
            self,
            dense_matrix: Optional[Union[np.ndarray, pd.DataFrame]]=None,
            data: Optional[np.ndarray]=None,
            indices: Optional[np.ndarray]=None,
            indptr: Optional[np.ndarray]=None,
            nrows: Optional[int]=None,
            ncols: Optional[int]=None,
            mappings_plus_inverses: Tuple[np.ndarray, np.ndarray]=None,
            fill_value: float=np.nan,
        ):

        self.mappings, self.mapped_inverses = np.unique(data, return_inverse=True) if mappings_plus_inverses is None else mappings_plus_inverses

        self._mappings = np.concatenate([np.array([fill_value]), ])

        self.data = []
        self.indices = []
        self.indptr = []
        self.shape = []

if __name__ == '__main__':
    original_data = np.array(list(reversed([1, 2, 3, 4, 5, 6, 7, 8, 9])))
    print(f'original_data = {original_data}')
    sparse = SparseMatrix(
        data=original_data,
        indices=np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        indptr=np.array([0, 3, 6, 9]),
        nrows=3,
        ncols=3,
    )
    print(f'{sparse.mappings=}')
    print(f'{sparse.mapped_inverses=}')
    reconstructed = sparse.mappings[sparse.mapped_inverses]
    print(f'{reconstructed=}')