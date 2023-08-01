from typing import DefaultDict, List, TypeVar, Optional, Union, List, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import matplotlib.pyplot as plt
import scipy.sparse

def alignmentInfoTrialPairs_to_signalColumns(
        df_trial: pd.DataFrame,
        lst_columnNamePairs_trial_alignInfo: List[Tuple[str, str]],
        num_rows: int,
        fillValues_absent: float=np.nan,
        fillValues_present_noInfo: float=1.0
    ) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a dataframe of trial table alignment information to a numpy array of signal columns.
    (Each row in the output dataframe corresponds to a row in df_signal dataframe.)
    (Each column in the output numpy array corresponds to a column in lst_columnNamePairs_trial_alignInfo.)
    (The output numpy array will have the same number of rows as df_signal_index and the same number of columns as lst_columnNamePairs_trial_alignInfo.)

    Args:
        df_trial (pd.DataFrame): Dataframe of trial table alignment columns and information columns
        lst_columnNamePairs_trial_alignInfo (List[Tuple[str, str]]): List of column name pairs. Each pair is a tuple of (trial table column name, alignment information column name).
        num_rows (int): Number of rows in the output numpy array

    Returns:
        arr_aligned (np.ndarray): Numpy array of signal columns
        lst_output_column_names (List[str]): List of column names associated with arr_aligned
    """

    lst_tmp = []
    lst_output_column_names = []

    for str_columnPair_alignment, str_columnPair_information in lst_columnNamePairs_trial_alignInfo:
        if str_columnPair_alignment == str_columnPair_information:
            output_column_name = str_columnPair_alignment
            fillValues_absent = fillValues_absent
            fillValues_present = fillValues_present_noInfo
        else:
            output_column_name = f"{str_columnPair_alignment}={str_columnPair_information}"
            fillValues_absent = fillValues_absent
            fillValues_present = df_trial[str_columnPair_information]
        

        tmp = np.full((num_rows), fillValues_absent).astype(object)
        tmp[df_trial[str_columnPair_alignment].values] = fillValues_present
        lst_tmp.append(tmp)
        lst_output_column_names.append(output_column_name)

    return np.stack(lst_tmp, axis=1), lst_output_column_names


def alignmentInfoTrialPairs_to_signalColumns_sparse(
        df_alignment: pd.DataFrame,
        df_info: pd.DataFrame,
        num_rows: int,
        include_alignment_as_info: bool = False,
        fillValues_absent: float=np.nan,
        fillValues_aligmentPresent: float=1.0,
    ) -> Tuple[scipy.sparse.coo_matrix, List[str]]:
    """
    Convert a dataframe of trial table alignment information to a sparse numpy array of signal columns.

    Args:
        df_alignment (pd.DataFrame): Dataframe of trial table alignment columns
        df_info (pd.DataFrame): Dataframe of trial table information columns
        num_rows (int): Number of rows in the output numpy array
        include_alignment_as_info (bool): Whether to include the alignment columns as onehot information columns
        fillValues_absent (float): Value to use for absent values
        fillValues_aligmentPresent (float): Value to use for alignment present values

    Returns:
        arr_aligned (scipy.sparse.coo_matrix): Sparse numpy array of signal columns
        lst_output_column_names (List[str]): List of column names associated with arr_aligned
    """
    lst_row = []
    lst_col = []
    lst_data = []
    lst_output_column_names = []

    if include_alignment_as_info:
        for inx_alignment in range(df_alignment.shape[1]):
            colInx = 0 if len(lst_col) == 0 else max(lst_col) + 1
            addl_rowInx = df_alignment.iloc[:, inx_alignment].values.tolist()
            addl_colInx = [colInx] * df_alignment.shape[0]
            addl_data = [fillValues_aligmentPresent] * df_alignment.shape[0]
            addl_column_names = [df_alignment.columns[inx_alignment]]

            lst_row.extend(addl_rowInx)
            lst_col.extend(addl_colInx)
            lst_data.extend(addl_data)
            lst_output_column_names.extend(addl_column_names)
    
    for inx_alignment in range(df_alignment.shape[1]):
        for inx_info in range(df_info.shape[1]):
            colInx = 0 if len(lst_col) == 0 else max(lst_col) + 1
            addl_rowInx = df_alignment.iloc[:, inx_alignment].values.tolist()
            addl_colInx = [colInx] * df_alignment.shape[0]
            addl_data = df_info.iloc[:, inx_info].values.tolist()
            addl_columns = [(df_alignment.columns[inx_alignment], df_info.columns[inx_info])]

            lst_row.extend(addl_rowInx)
            lst_col.extend(addl_colInx)
            lst_data.extend(addl_data)
            lst_output_column_names.extend(addl_columns)
    
    arr_aligned_info = scipy.sparse.coo_matrix((lst_data, (lst_row, lst_col)), shape=(num_rows, max(lst_col) + 1))
    arr_aligned_info.fill_value = fillValues_absent

    return (arr_aligned_info,
           lst_output_column_names)

if __name__ == "__main__":
    print('Testing alignmentInfoTrialPairs_to_signalColumns_sparse()')
    df_alignment = pd.DataFrame({
        'a': [0, 1, 2, 3],
        'b': [1, 2, 3, 4],
    })
    df_info = pd.DataFrame({
        'c': [0, 1, 2, 3],
        'd': [1, 2, 3, 4],
    })
    num_rows = 10
    arr_aligned, lst_output_column_names = alignmentInfoTrialPairs_to_signalColumns_sparse(df_alignment, df_info, num_rows)
    assert np.all(pd.DataFrame(arr_aligned.toarray(), columns=lst_output_column_names) == pd.DataFrame(
        {
            'a': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            'b': [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            ('a','c'): [0, 1, 2, 3, 0, 0, 0, 0, 0, 0],
            ('a','d'): [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
            ('b','c'): [0, 0, 1, 2, 3, 0, 0, 0, 0, 0],
            ('b','d'): [0, 1, 2, 3, 4, 0, 0, 0, 0, 0],
        }
        )
    )

def add_column_nTrial_raw(df: pd.DataFrame, columnName_trialId: str='nTrial_raw'):
    """
    Create a column of trial numbers (0-indexed) for each row in df.
    If columnName_trialId is specified, it will be used as the name of the new column.
    Otherwise, the column will be named "nTrial_raw".
    (columnName_trialId must not already exist in df.)

    JZ 2023

    Args:
        df (pd.DataFrame): Dataframe to add trial number column to
        columnName_trialId (Optional[str]): Name of the new column

    Returns:
        df (pd.DataFrame): Dataframe with new column
    """

    assert columnName_trialId not in df.columns, f"{columnName_trialId} already exists in df. Please remove/rename {columnName_trialId} in df or specify a different columnName_trialId."
    return df.assign(
        **{columnName_trialId: pd.Series(np.arange(df.shape[0]), index=df.index, name=columnName_trialId)}
    )


def check_monotonically_increasing_df(df: pd.DataFrame, lst_ignoreRowsWithValues: List[Any] = [-1, 0], bool_ignoreRowsWithNans: bool = True, equal_allowed: bool = False):
    """
    Check that all values in the dataframe are monotonically increasing.
    This is a requirement for the trial table alignment values.
    (Compares the max alignment value of each row to the min of the next row.)

    JZ 2023

    Args:
        df (pd.DataFrame): Dataframe to check
    """
    df = df[~(df.isin(lst_ignoreRowsWithValues).any(axis=1))]
    if bool_ignoreRowsWithNans:
        df = df[~df.isna().any(axis=1)]
    if equal_allowed:
        assert np.all(df.iloc[:-1].max(axis=1).values <= df.iloc[1:].min(axis=1).values), "Trial table alignment values must be monotonically non-decreasing for non-zero entries."
    else:
        assert np.all(df.iloc[:-1].max(axis=1).values < df.iloc[1:].min(axis=1).values), "Trial table alignment values must be monotonically increasing for non-zero entries."

