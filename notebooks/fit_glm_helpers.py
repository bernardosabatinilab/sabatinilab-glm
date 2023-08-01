from typing import DefaultDict, List, TypeVar, Optional, Union, List, Tuple
import pandas as pd
import numpy as np


def check_monotonically_increasing_df(df: pd.DataFrame, bool_ignoreRowsWithZeros: bool = True, bool_ignoreRowsWithNans: bool = True, equal_allowed: bool = False):
    """
    Check that all values in the dataframe are monotonically increasing.
    This is a requirement for the trial table alignment values.
    (Compares the max alignment value of each row to the min of the next row.)

    JZ 2023

    Args:
        df (pd.DataFrame): Dataframe to check
    """
    if bool_ignoreRowsWithZeros:
        df = df[~(df==0).any(axis=1)]
    
    if bool_ignoreRowsWithNans:
        df = df[~df.isna().any(axis=1)]
    
    if equal_allowed:
        assert np.all(df.iloc[:-1].max(axis=1).values <= df.iloc[1:].min(axis=1).values), "Trial table alignment values must be monotonically non-decreasing for non-zero entries."
    else:
        assert np.all(df.iloc[:-1].max(axis=1).values < df.iloc[1:].min(axis=1).values), "Trial table alignment values must be monotonically increasing for non-zero entries."

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

def alignmentInfoTrialPairs_to_signalColumns(df_trial: pd.DataFrame, lst_columnNamePairs_trial_alignInfo: List[Tuple[str, str]], df_signal_index: pd.Index):
    """
    Convert a dataframe of trial table alignment information to a dataframe of signal columns.
    (Each row in the output dataframe corresponds to a row in df_signal dataframe.)
    (Each column in the output dataframe corresponds to a column in lst_columnNamePairs_trial_alignInfo.)
    (The output dataframe will have the same number of rows as df_signal_index and the same number of columns as lst_columnNamePairs_trial_alignInfo.)
    
    JZ 2023

    Args:
        df_trial (pd.DataFrame): Dataframe of trial table alignment columns and information columns
        lst_columnNamePairs_trial_alignInfo (List[Tuple[str, str]]): List of column name pairs. Each pair is a tuple of (trial table column name, alignment information column name).
        df_signal_index (pd.Index): Index of the output dataframe

    Returns:
        df (pd.DataFrame): Dataframe of signal columns
    """

    lst_tmp = [] 
    lst_output_column_names = []

    for str_columnPair_alignment, str_columnPair_information in lst_columnNamePairs_trial_alignInfo:
        if str_columnPair_alignment == str_columnPair_information:
            output_column_name = str_columnPair_alignment
            fillValues_absent = np.nan
            fillValues_present = 1
        else:
            output_column_name = f"{str_columnPair_alignment}={str_columnPair_information}"
            fillValues_absent = np.nan
            fillValues_present = df_trial[str_columnPair_information]
        

        tmp = np.full((len(df_signal_index)), fillValues_absent).astype(object)
        tmp[df_trial[str_columnPair_alignment].values] = fillValues_present
        lst_tmp.append(tmp)
        lst_output_column_names.append(output_column_name)

    return pd.DataFrame(np.stack(lst_tmp, axis=1), columns=lst_output_column_names, index=df_signal_index)
