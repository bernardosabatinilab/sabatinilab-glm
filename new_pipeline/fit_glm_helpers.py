from typing import DefaultDict, List, TypeVar, Optional, Union, List, Tuple, Dict
import numpy as np
import pandas as pd
import scipy.sparse




def construct_vector_or_matrix(
        name: str,
        values: Optional[np.ndarray]=None,
        indices: Optional[np.ndarray]=None,
        nrows: Optional[int]=None,
        fill_values: Optional[float]=0, # Not yet implemented: Non-zero fill-values
        dtype: Optional[type]=None, # Use 'categorical' for categorical values # Non-categorical values are numerical # Other types not yet implemented
    ):

    assert (values is not None) or (indices is not None), 'At least one of values or indices must be specified.'
    assert fill_values == 0, 'Non-zero fill-values not yet implemented.'

    indices_arangeFilled = indices if indices is not None else np.arange(len(values))
    dtype = dtype if dtype is not None else 'numerical' # self.
    nrows = nrows if nrows is not None else np.max(indices_arangeFilled) + 1 # self.

    if dtype != 'categorical': # self.
        name_tuple = (name,) # self.
        # subvectors = {} # self.
        # subvectors[name] = Vector(
        #     name_tuple=name_tuple,
        #     values=values,
        #     indices=indices,
        #     nrows=nrows,
        #     fill_values=fill_values,
        #     dtype='numerical',
        # )
        # return Matrix(**subvectors)
        return Matrix(
            Vector(
                name_tuple=name_tuple,
                values=values,
                indices=indices,
                nrows=nrows,
                fill_values=fill_values,
                dtype='numerical',
            )
        )
    else:
        categories_unique = np.unique(values)
        # subvectors = {} # self.
        subvectors = []
        for category in categories_unique:
            name_tuple = (name, category)
            subvector = Vector(
                name_tuple=name_tuple,
                values=None,
                indices=indices_arangeFilled[values == category],
                nrows=nrows,
                fill_values=fill_values,
                dtype='numerical',
            )
            # subvectors[(name, category)] = subvector # self.
            subvectors.append(subvector)
        return Matrix(*subvectors)



# Needed: Values and/or indices, but at least one. Specification of values as categorical or numerical. nrows. fill_values.
class Vector():
    def __init__(
            self,
            name_tuple: Tuple[str],
            values: Optional[np.ndarray]=None,
            indices: Optional[np.ndarray]=None,
            nrows: Optional[int]=None,
            fill_values: Optional[float]=None, # Not yet implemented: Non-zero fill-values
            dtype: Optional[type]=None, # Use 'categorical' for categorical values # Non-categorical values are numerical # Other types not yet implemented
        ):

        assert (values is not None) or (indices is not None), 'At least one of values or indices must be specified.'
        assert fill_values == 0 or fill_values is None, 'Non-zero fill-values not yet implemented.'
        assert dtype == 'numerical' or dtype is None, 'Vectors must be of dtype numerical. If categorical, create a Matrix via construct_vector_or_matrix.'

        self.name_tuple = name_tuple
        self.values = values
        self.indices = indices
        indices_arangeFilled = indices if indices is not None else np.arange(len(values))
        self.nrows = nrows if nrows is not None else np.max(indices_arangeFilled) + 1
        self.fill_values = fill_values if fill_values is not None else 0
        self.dtype = dtype if dtype is not None else 'numerical'
        self.ncols = 1

        assert self.values is None or self.indices is None or len(self.values) == len(self.indices), 'Values and indices must be of the same length.'

    def __repr__(self):
        return f'Vector(dtype={self.dtype}, nrows={self.nrows}, values={self.values}, indices={self.indices})'
    
    def __str__(self):
        return self.__repr__()
    
    def __and__(self, other: 'Vector'):

        assert self.values is None or other.values is None, 'Only one of the subcomponents can have values <> None.'

        set_nrows = set([self.nrows, other.nrows])
        assert len(set_nrows) == 1, 'All subcomponents must have the same number of nrows.'
        nrows = list(set_nrows)[0]

        set_fill_values = set([self.fill_values, other.fill_values])
        assert len(set_fill_values) == 1, 'All subcomponents must have the same fill values.'
        fill_values = list(set_fill_values)[0]
        
        if self.values is not None:
            values = self.values
            indices_values = self.indices
            indices_other = other.indices
        elif other.values is not None:
            values = other.values
            indices_values = other.indices
            indices_other = self.indices
        else:
            values = None
            indices_values = self.indices
            indices_other = other.indices

            ##### TODO: PICK UP HERE -- IMPLEMENTING THE TRANSITION FROM VALUES / INDICES TO SITUATIONS WHERE ONE OR OTHER NOT SPECIFIED
        
        
        if indices_values is None and indices_other is None:
            indices = None
        else:
            indices_intersected = np.arange(self.nrows)
            if indices_values is not None:
                indices_intersected = np.intersect1d(indices_intersected, indices_values)
            if other.indices is not None:
                indices_intersected = np.intersect1d(indices_intersected, indices_other)
            
            values, indices = values[np.isin(indices_values, indices_intersected)], indices_values[np.isin(indices_values, indices_intersected)]

        name_tuple = (self.name_tuple, other.name_tuple)
        return Vector(
            name_tuple=name_tuple,
            values=values,
            indices=indices,
            nrows=nrows,
            fill_values=fill_values, # Not yet implemented: Non-zero fill-value
            dtype='numerical',
        )
    
    def todense(self):
        values = np.zeros(self.nrows)
        indices_arangeFilled = self.indices if self.indices is not None else np.arange(len(values))
        values[indices_arangeFilled] = self.values if self.values is not None else 1
        return values
    
    def topd(self):
        return pd.Series(self.todense(), name=self.name_tuple)


class Matrix():
    def __init__(
            self,
            *list_subcomponents: List[Vector],
            **dict_subcomponents: Dict[str, Vector],
        ):
        dict_subcomponents_from_list = {subcomponent.name_tuple: subcomponent for subcomponent in list_subcomponents}
        assert len(dict_subcomponents_from_list) == len(list_subcomponents), 'All subcomponents must have unique names.'
        subcomponents = dict_subcomponents_from_list | dict_subcomponents
        assert len(subcomponents) == len(list_subcomponents) + len(dict_subcomponents), 'All subcomponents must have unique names.'
        set_nrows = set([component.nrows for component in subcomponents.values()])
        assert len(set_nrows) == 1, 'All vectors in a matrix must have the same number of nrows.'
        for key_subcomponent, subcomponent in subcomponents.items():
            assert isinstance(subcomponent, Vector), 'All subcomponents must be of type Vector.'
        self.subcomponents = subcomponents

        self.ncols = sum([subcomponent.ncols for subcomponent in subcomponents.values()])
        self.nrows = list(set_nrows)[0]

    def __repr__(self):
        return f'Matrix(nrows={self.nrows}, ncols={self.ncols}, subcomponents={self.subcomponents})'
    
    def __str__(self):
        return self.__repr__()
    
    def __iter__(self):
        return iter(self.subcomponents.values())
    
    def __getitem__(self, key):
        assert not isinstance(key, slice), 'Slicing not allowed for Matrices.'

        # If the get item key type is a list
        if isinstance(key, list):
            return Matrix(*[self[single_key] for single_key in key])
        
        elif key in self.subcomponents:
            return self.subcomponents[key]
        
        elif isinstance(key, str) and (key,) in self.subcomponents:
            return self.subcomponents[(key,)]
        
        # If the key type is a tuple return an intersection of the subcomponents' indices where only one subcomponent is allowed to have a non-none value
        elif isinstance(key, tuple) and len(key) > 1:
            subcomponents_queried = [self[single_key] for single_key in key]

            # Create a variable to represent the values of the single (or zero) subcomponent that has non-none values
            lst_subcomponents_with_values = [subcomponent for subcomponent in subcomponents_queried if subcomponent.values is not None]

            # Assert values is not none for only one subcomponent
            assert len(lst_subcomponents_with_values) <= 1, 'A maximum of one subcomponent can have values <> None.'
            
            set_nrows = set([subcomponent.nrows for subcomponent in subcomponents_queried])
            assert len(set_nrows) == 1, 'All subcomponents must have the same number of nrows.'

            set_fill_values = set([subcomponent.fill_values for subcomponent in subcomponents_queried])
            assert len(set_fill_values) == 1, 'All subcomponents must have the same fill values.'
            
            # Create a list of all subcomponent indices
            lst_indices = [subcomponent.indices if subcomponent.indices is not None else np.arange(subcomponent.nrows) for subcomponent in subcomponents_queried]

            # Filter all indices and values to only include those that are in the intersection of the subcomponent indices
            for i, indices in enumerate(lst_indices):
                indices_intersected = indices if i == 0 else np.intersect1d(indices_intersected, indices)
            
            values_indices_intersected = [(subcomponent.values[np.isin(subcomponent.indices, indices_intersected)],
                                           subcomponent.indices[np.isin(subcomponent.indices, indices_intersected)]
                                          ) for subcomponent in lst_subcomponents_with_values]
            
            values_intersected = [subcomponent[0] for subcomponent in values_indices_intersected]
            
            indices_intersected = [subcomponent[1] for subcomponent in values_indices_intersected]


            name_tuple = tuple([subcomponent.name_tuple for subcomponent in subcomponents_queried])
            return Vector(
                name_tuple=name_tuple,
                values=values_intersected[0],
                indices=indices_intersected[0],
                nrows=list(set_nrows)[0],
                fill_values=list(set_fill_values)[0], # Not yet implemented: Non-zero fill-value
                dtype='numerical',
            )
        else:
            raise KeyError('Key must be a single str found in Matrix key, tuple comprised of keys found in Matrix, or list thereof.')
    
    def todense(self):
        values = np.zeros((self.nrows, self.ncols))
        for i, subcomponent in enumerate(self.subcomponents.values()):
            values[:, i] = subcomponent.todense()
        return values
    
    def topd(self):
        return pd.DataFrame(self.todense(), columns=list(self.subcomponents.keys()))


if __name__ == '__main__':
    # Test Vector
    vec = Vector(
        name_tuple=('a',),
        values=np.array([1, 2, 3]),
        indices=np.array([0, 1, 2]),
        nrows=5,
        fill_values=0,
        dtype='numerical',
    )
    print(vec)

    # Test Matrix
    mat = Matrix(
        vec,
        Vector(
            name_tuple=('a', 'b'),
            values=np.array([4, 5, 6]),
            indices=np.array([1, 2, 3]),
            nrows=5,
            fill_values=0,
            dtype='numerical',
        ),
        Vector(
            name_tuple=('b', 'c'),
            values=np.array([7, 8, 9]),
            indices=np.array([2, 3, 4]),
            nrows=5,
            fill_values=0,
            dtype='numerical',
        ),
        Vector(
            name_tuple=('d',),
            indices=np.array([2, 3]),
            nrows=5,
            fill_values=0,
            dtype='numerical',
        )
    )
    print(mat.todense())
    print(mat.topd())
    print(mat['a',].todense())
    print(mat['a',].topd())
    print(mat['a', 'b'].todense())
    print(mat['a', 'b'].topd())
    print(mat['b', 'c'].todense())
    print(mat['b', 'c'].topd())
    print(mat[['a', 'd']].todense())
    print(mat[['a', 'd']].topd())
    print(mat[[('a', 'd')]].todense())
    print(mat[[('a', 'd')]].topd())


    # Test Vector Categorical
    cat = construct_vector_or_matrix(
        name='category',
        values=np.array([0, 1, 'a', 'a', 1]),
        indices=np.array([1, 2, 5, 6, 8]),
        nrows=9,
        fill_values=0,
        dtype='categorical',
    )
    print(cat)
    print(cat.todense())
    print(cat.topd())

    print()

# Dense matrix constructed from just values (DV) gets associated with arange implied indices


# Sparse matrix constructed from just indices (SI) gets ones associated with specified indices and non-specified indices get zeros otherwise within shape of rows


# Sparse matrix constructed from both indices and values (SIV) gets values associated with indices and non-specified indices get zeros otherwise within shape of rows


# Dense categorical values (DC) are subsegmented into sparse matrices with each segmenttion of categories being indicated by the unique values of the categorical variable 


# Sparse categorical values (SC) are subsegmented into sparse matrices with each segmenttion of categories being indicated by the unique values of the categorical variable (except value 0 being unassociated)



# class SparseMatrix():
#     """
#     A class for storing sparse matrices.

#     JZ 2023
#     """
#     def __init__(
#             self,
#             dense_matrix: Optional[Union[np.ndarray, pd.DataFrame]]=None,
#             data: Optional[np.ndarray]=None,
#             indices: Optional[np.ndarray]=None,
#             indptr: Optional[np.ndarray]=None,
#             nrows: Optional[int]=None,
#             ncols: Optional[int]=None,
#             mappings_plus_inverses: Tuple[np.ndarray, np.ndarray]=None,
#             fill_value: float=np.nan,
#         ):

#         self.mappings, self.mapped_inverses = np.unique(data, return_inverse=True) if mappings_plus_inverses is None else mappings_plus_inverses

#         self._mappings = np.concatenate([np.array([fill_value]), ])

#         self.data = []
#         self.indices = []
#         self.indptr = []
#         self.shape = []

# if __name__ == '__main__':
#     original_data = np.array(list(reversed([1, 2, 3, 4, 5, 6, 7, 8, 9])))
#     print(f'original_data = {original_data}')
#     sparse = SparseMatrix(
#         data=original_data,
#         indices=np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
#         indptr=np.array([0, 3, 6, 9]),
#         nrows=3,
#         ncols=3,
#     )
#     print(f'{sparse.mappings=}')
#     print(f'{sparse.mapped_inverses=}')
#     reconstructed = sparse.mappings[sparse.mapped_inverses]
#     print(f'{reconstructed=}')