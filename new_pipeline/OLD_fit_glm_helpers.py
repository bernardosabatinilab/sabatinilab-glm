from typing import DefaultDict, List, TypeVar, Optional, Union, List, Tuple, Dict
import numpy as np
import pandas as pd
import scipy.sparse


# From onehot to value / index
if __name__ == '__main__':
    arr = np.array([1, 0, 0, 0, 1, 0, 0])
    # print(np.unique(arr, return_inverse=True))
    print(np.arange(len(arr))[arr != 0])

def densified_sparse_to_sparse(densified_sparse, fill_value=0):
    nrows = densified_sparse.shape[0]
    sparse_indices = np.arange(nrows)[densified_sparse != fill_value]
    sparse_values = densified_sparse[sparse_indices]

    unq_sparse_values = np.unique(sparse_values)
    if len(unq_sparse_values) == 1 and unq_sparse_values[0] == 1:
        sparse_values = None

    return {'indices': sparse_indices, 'values': sparse_values, 'nrows': nrows}

# From onehot to value / index
if __name__ == '__main__':
    arr = np.array([1, 0, 0, 0, 1, 0, 0])
    print(densified_sparse_to_sparse(np.array([1, 0, 0, 0, 1, 0, 0]), fill_value=0))


def construct_vector_or_matrix(
        name: str,
        values: Optional[np.ndarray]=None,
        indices: Optional[np.ndarray]=None,
        nrows: Optional[int]=None,
        fill_values: Optional[float]=0, # Not yet implemented: Non-zero fill-values
        dtype: Optional[type]=None, # Use 'categorical' for categorical values # Non-categorical values are numerical # Other types not yet implemented
    ):

    assert (values is not None) or (indices is not None), 'At least one of values or indices must be specified.'
    # assert fill_values == 0, 'Non-zero fill-values not yet implemented.'

    indices_arangeFilled = indices if indices is not None else np.arange(len(values))
    dtype = dtype if dtype is not None else 'numerical' # self.
    nrows = nrows if nrows is not None else np.max(indices_arangeFilled) + 1 # self.

    if dtype != 'categorical':
        name_tuple = (name,)
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
            name_tuple: Union[str, Tuple[str]],
            values: Optional[np.ndarray]=None,
            indices: Optional[np.ndarray]=None,
            nrows: Optional[int]=None,
            fill_values: Optional[float]=None, # Not yet implemented: Non-zero fill-values
            dtype: Optional[type]=None, # Use 'categorical' for categorical values # Non-categorical values are numerical # Other types not yet implemented
        ):

        assert (values is not None) or (indices is not None), 'At least one of values or indices must be specified.'
        # assert fill_values == 0 or fill_values is None, 'Non-zero fill-values not yet implemented.'
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
    
    def __and__(self, other: Union['Vector', 'Matrix']):
        
        if isinstance(other, Vector):
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
        elif isinstance(other, Matrix):
            return other & self
        else:
            raise NotImplementedError(f'__and__ not implemented for type {type(other)}.')
    
    def todense(self):
        values = np.full(self.nrows, self.fill_values)
        indices_arangeFilled = self.indices if self.indices is not None else np.arange(len(values))
        values[indices_arangeFilled] = self.values if self.values is not None else 1
        return values
    
    def topd(self):
        return pd.Series(self.todense(), name=self.name_tuple)
    
    def shift(self, n: int):
        indices = (np.arange(self.nrows) if self.indices is None else self.indices) + n
        values = self.values[(indices >= 0)&(indices < self.nrows)] if self.values is not None else None
        indices = indices[(indices >= 0)&(indices < self.nrows)]
        return Vector(
            name_tuple=tuple(list(self.name_tuple) + [f'shift_{n}']),
            values=values,
            indices=indices,
            nrows=self.nrows,
            fill_values=self.fill_values,
            dtype=self.dtype,
        )
    
    def shift_multiple(self, ns: List[int]):
        return Matrix(*[self.shift(n=n) for n in ns])


class Matrix():
    def __init__(
            self,
            *list_subcomponents: List[Union[Vector, 'Matrix']],
            **dict_subcomponents: Dict[str, Union[Vector, 'Matrix']],
        ):

        list_subcomponents_flattened = []
        for subcomponent in list_subcomponents:
            if isinstance(subcomponent, Vector):
                list_subcomponents_flattened.append(subcomponent)
            elif isinstance(subcomponent, Matrix):
                list_subcomponents_flattened += list(subcomponent.subcomponents.values())
            else:
                raise NotImplementedError(f'__init__ not implemented for type {type(subcomponent)}.')
        
        dict_subcomponents_flattened = {}
        for name_subcomponent, subcomponent in dict_subcomponents.items():
            if isinstance(subcomponent, Vector):
                dict_subcomponents_flattened[name_subcomponent] = subcomponent
            elif isinstance(subcomponent, Matrix):
                for name_subsubcomponent, subsubcomponent in subcomponent.subcomponents.items():
                    dict_subcomponents_flattened[(name_subcomponent, name_subsubcomponent)] = subcomponent.subcomponents.values()
            else:
                raise NotImplementedError(f'__init__ not implemented for type {type(subcomponent)}.')

        dict_subcomponents_from_list = {subcomponent.name_tuple: subcomponent for subcomponent in list_subcomponents_flattened if isinstance(subcomponent, Vector)}
        dict_subcomponents_from_list = dict_subcomponents_from_list | {k_submatrix: v_submatrix for subcomponent in list_subcomponents_flattened if isinstance(subcomponent, Matrix) for k_submatrix, v_submatrix in subcomponent.subcomponents.items()}
        assert len(dict_subcomponents_from_list) == sum([subcomponent.ncols for subcomponent in list_subcomponents_flattened]), 'All subcomponents must have unique names.'
        subcomponents = dict_subcomponents_from_list | dict_subcomponents_flattened
        set_nrows = set([component.nrows for component in subcomponents.values()])
        assert len(set_nrows) == 1, 'All vectors in a matrix must have the same number of nrows.'
        for key_subcomponent, subcomponent in subcomponents.items():
            assert isinstance(subcomponent, Vector), 'All subcomponents must be of type Vector.'
        self.subcomponents = subcomponents

        set_fill_values = set([subcomponent.fill_values for subcomponent in self.subcomponents.values()])
        assert len(set_fill_values) == 1, 'All subcomponents must have the same fill values.'
        self.fill_values = list(set_fill_values)[0]

        self.ncols = sum([subcomponent.ncols for subcomponent in subcomponents.values()])
        self.nrows = list(set_nrows)[0]

    def __repr__(self):
        return f'Matrix(nrows={self.nrows}, ncols={self.ncols}, subcomponents={self.subcomponents})'
    
    def __str__(self):
        return self.__repr__()
    
    def __iter__(self):
        return iter(self.subcomponents.values())
    
    def __and__(self, other):
        if isinstance(other, Vector):
            return Matrix(
                *[self_subcomponent & other for self_subcomponent in self.subcomponents.values()]
            )
        elif isinstance(other, Matrix):
            return other & self
        else:
            raise NotImplementedError(f'__and__ not implemented for type {type(other)}.')
    
    def __getitem__(self, key):
        # List of keys will return a Matrix selected by each of the keys
        # Single key will return a Vector selected by the key
        # Key as a string will return a Matrix with all subcomponents that have that name as the first tuple entry
        # Key as a tuple will return a Vector selected very specifically by the entire key
        assert not isinstance(key, slice), 'Slicing not allowed for Matrices.'
        
        # If the get item key type is a list
        if isinstance(key, list):
            return Matrix(*[self[single_key] for single_key in key])
        
        elif (isinstance(key, str) or isinstance(key, tuple)) and key in self.subcomponents:
            return self.subcomponents[key]
        
        elif (isinstance(key, str) or isinstance(key, tuple)):
            all_with_key = [subcomponent for subcomponent in self.subcomponents.values() if
                               (isinstance(subcomponent.name_tuple, tuple) and key == subcomponent.name_tuple[0])
                            or (isinstance(subcomponent.name_tuple, str) and key == subcomponent.name_tuple)]
            
            if len(all_with_key) == 0:
                raise KeyError(f'No subcomponents found with the first key entry: "{key}". The only subcomponents are: {[subcomponent.name_tuple for subcomponent in self.subcomponents.values()]}')
            elif len(all_with_key) == 1:
                return all_with_key[0]
            else:
                return Matrix(*all_with_key)
        
        else:
            raise KeyError('Key must be a single str or key found in Matrix or a list thereof.')
    
    def todense(self):
        values = np.full((self.nrows, self.ncols), self.fill_values)
        for i, subcomponent in enumerate(self.subcomponents.values()):
            values[:, i] = subcomponent.todense()
        return values
    
    def topd(self):
        return pd.DataFrame(self.todense(), columns=list(self.subcomponents.keys()))

    def shift(self, n: int):
        return Matrix(*[subcomponent.shift(n=n) for subcomponent in self.subcomponents.values()])
    
    def shift_multiple(self, ns: List[int]):
        return Matrix(*[subcomponent.shift_multiple(ns=ns) for subcomponent in self.subcomponents.values()])


# if __name__ == '__main__':
#     # Test Vector
#     vec = Vector(
#         name_tuple=('vec_test',),
#         values=np.array([1, 2, 3]),
#         indices=np.array([0, 1, 2]),
#         nrows=9,
#         fill_values=0,
#         dtype='numerical',
#     )
#     print(vec)

#     # Test Matrix
#     mat = Matrix(
#         Vector(
#             name_tuple=('a', 'b'),
#             values=np.array([4, 5, 6]),
#             indices=np.array([1, 2, 3]),
#             nrows=9,
#             fill_values=0,
#             dtype='numerical',
#         ),
#         Vector(
#             name_tuple=('b', 'c'),
#             values=np.array([7, 8, 9]),
#             indices=np.array([2, 3, 4]),
#             nrows=9,
#             fill_values=0,
#             dtype='numerical',
#         ),
#         Vector(
#             name_tuple=('d',),
#             indices=np.array([2, 3]),
#             nrows=9,
#             fill_values=0,
#             dtype='numerical',
#         )
#     )
#     print(mat.todense())
#     print(mat.topd())
#     print(mat['a'].todense())
#     print(mat['a'].topd())
#     print(mat['a', 'b'].todense())
#     print(mat['a', 'b'].topd())
#     print(mat['b', 'c'].todense())
#     print(mat['b', 'c'].topd())
#     print(mat[['a', 'd']].todense())
#     print(mat[['a', 'd']].topd())

#     print()

#     # Test Vector Categorical
#     cat = construct_vector_or_matrix(
#         name='category',
#         values=np.array([0, 1, 'a', 'a', 1]),
#         indices=np.array([1, 2, 5, 6, 8]),
#         nrows=9,
#         fill_values=0,
#         dtype='categorical',
#     )
#     print(cat)
#     print(cat.todense())
#     print(cat.topd())
#     try:
#         print(cat['a'].topd())
#         assert False, 'Should not be able to select a category from a categorical vector.'
#     except KeyError:
#         pass
#     print(cat['category'].topd())

#     combined_mat_cat = Matrix(
#         vec,
#         mat,
#         cat,
#     )
#     print('Matrix Setup')
#     print(combined_mat_cat)
#     print(combined_mat_cat.todense())
#     print(combined_mat_cat.topd())


#     cat_subselected = combined_mat_cat['category']
#     cat_0_subselected = combined_mat_cat[('category', '0')]
#     vec_subselected = combined_mat_cat['vec_test']
#     print('ampersand test')
#     print(cat_subselected.topd())
#     print(cat_0_subselected.topd())
#     print(vec_subselected.topd())
#     print((vec_subselected & cat_0_subselected).topd())
#     print((vec_subselected & cat_subselected).topd())
#     print((cat_0_subselected & vec_subselected).topd())
#     print((cat_subselected & vec_subselected).topd())

#     print('shift test')
#     print(cat_subselected.shift(-2).topd())
#     print(cat_subselected.shift(2).topd())
#     print(cat_0_subselected.shift(-2).topd())
#     print(cat_0_subselected.shift(2).topd())
#     print(vec_subselected.shift(-2).topd())
#     print(vec_subselected.shift(2).topd())

#     print(cat_subselected.shift_multiple([-2, 0, 2]).topd())
#     print(cat_0_subselected.shift_multiple([-2, 0, 2]).topd())
#     print(vec_subselected.shift_multiple([-2, 0, 2]).topd())

#     print()

# # Dense matrix constructed from just values (DV) gets associated with arange implied indices


# # Sparse matrix constructed from just indices (SI) gets ones associated with specified indices and non-specified indices get zeros otherwise within shape of rows


# # Sparse matrix constructed from both indices and values (SIV) gets values associated with indices and non-specified indices get zeros otherwise within shape of rows


# # Dense categorical values (DC) are subsegmented into sparse matrices with each segmenttion of categories being indicated by the unique values of the categorical variable 


# # Sparse categorical values (SC) are subsegmented into sparse matrices with each segmenttion of categories being indicated by the unique values of the categorical variable (except value 0 being unassociated)



# # class SparseMatrix():
# #     """
# #     A class for storing sparse matrices.

# #     JZ 2023
# #     """
# #     def __init__(
# #             self,
# #             dense_matrix: Optional[Union[np.ndarray, pd.DataFrame]]=None,
# #             data: Optional[np.ndarray]=None,
# #             indices: Optional[np.ndarray]=None,
# #             indptr: Optional[np.ndarray]=None,
# #             nrows: Optional[int]=None,
# #             ncols: Optional[int]=None,
# #             mappings_plus_inverses: Tuple[np.ndarray, np.ndarray]=None,
# #             fill_value: float=np.nan,
# #         ):

# #         self.mappings, self.mapped_inverses = np.unique(data, return_inverse=True) if mappings_plus_inverses is None else mappings_plus_inverses

# #         self._mappings = np.concatenate([np.array([fill_value]), ])

# #         self.data = []
# #         self.indices = []
# #         self.indptr = []
# #         self.shape = []

# # if __name__ == '__main__':
# #     original_data = np.array(list(reversed([1, 2, 3, 4, 5, 6, 7, 8, 9])))
# #     print(f'original_data = {original_data}')
# #     sparse = SparseMatrix(
# #         data=original_data,
# #         indices=np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
# #         indptr=np.array([0, 3, 6, 9]),
# #         nrows=3,
# #         ncols=3,
# #     )
# #     print(f'{sparse.mappings=}')
# #     print(f'{sparse.mapped_inverses=}')
# #     reconstructed = sparse.mappings[sparse.mapped_inverses]
# #     print(f'{reconstructed=}')