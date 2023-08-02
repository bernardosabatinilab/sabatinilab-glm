from typing import Dict, List, TypeVar, Optional, Union, List, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import matplotlib.pyplot as plt
import util

class TrialPreprocessor():
    """
    Processor to handle trial table data inlcuding:
    * Dropping dummy alignment values
    * Handle matlab indexing to convert to python indexing
    * Assert that the alignment indices do not go below 0
    * Assert that the alignment indices are monotonically increasing
    * Create a column for unique trial ids within file it does not already exist

    JZ 2023

    Args:
        - df_trial (pd.DataFrame): Dataframe of trial table alignment and information columns.
        - sessionId (str): Session id to be used for all trials. If ``None``, a
            session id will be generated for each file.
        - column_trialId (str): Name of column containing trial ids.
          If doesn't exist in df_trial, a new column with values 0-N will be created.
            (Default is ``'nTrial_raw'``)
        - column_trial_start (str): Name of alignment column to identify trial start.
            If ``None`` and ``column_trial_end`` is not ``None``, the value used will be the same
            as ``column_trial_end``. At least one must be specified. (Default is ``None``)
        - column_trial_end (str):  Name of alignment column to identify trial end.
            If ``None`` and ``column_trial_start`` is not ``None``, the value used will be the
            same as ``column_trial_start``. At least one must be specified. (Default is ``None``)
        - columns_alignment (Optional[List[str]]): List of alignment column names. (Alignment
            columns take on integer values associated with the index at which an event
            occurs in the signal table.) If ``None``, all columns in df_trial will be
            considered alignment columns. (Default is ``None``)
        - columns_information_point (Optional[List[str]]): List of information column names
            that should be treated as point information (i.e. the value should only be associated
            with the specific alignment value assigned). If ``None``, no columns in df_trial will
            be treated as point information. (Default is ``None``)
        - columns_information_broadcast (Optional[List[str]]): List of information column names
            that should be treated as broadcast information (i.e. the value should be broadcast
            to all timestamps of a given trial). If ``None``, no columns in df_trial will be
            treated as broadcast information. (Default is ``None``)
        - matlab_indexing (bool): Whether the alignment columns are indexed with
            matlab indexing (i.e. 1-indexed) or python indexing (i.e. 0-indexed). (Default is
            ``True``)
        - alignment_ignoreValues (Optional[List[Union[int, float]]]): Value that if found in an
            alignment column should be considered a dummy value. Rows with dummy values will be
            dropped. If ``None``, value will default to 0 for matlab indexing and -1 for python
            indexing. (Default is ``None``)
        - dict_column_drop_values (Optional[Dict[str, List]]): Dictionary of column names and
            values that should be dropped. (Default is ``None``)
        - fillValue_sparse (Tuple[type, Any]): Tuple of type and value with which to fill sparse
            information columns. (Default is ``(float, 0.0)``)

    Attributes:
        - df_trial_raw (pd.DataFrame): Dataframe of trial table alignment and information columns
        - column_trialId (Optional[str]): Name of column containing trial ids.
        - column_trial_start (Optional[str]): Name of alignment column to identify trial start.
        - column_trial_end (Optional[str]): Name of alignment column to identify trial start.
        - columns_alignment (Optional[List[str]]): List of alignment column names.
        - columns_information_point (Optional[List[str]]): List of information column names
            that should be treated as point information.
        - columns_information_broadcast (Optional[List[str]]): List of information column names
            that should be treated as broadcast information.
        - matlab_indexing (Optional[bool]): Whether the alignment columns are indexed with
            matlab indexing (i.e. 1-indexed) or python indexing (i.e. 0-indexed).
        - alignment_ignoreValues (Optional[List[Union[int, float]]]): Value that if found in an
            alignment column should be considered a dummy value. Rows with dummy values will be
            dropped.
        - dict_column_drop_values (Optional[Dict[str, List]]): Dictionary of column names and
            values that should be dropped.
        - df_trial (pd.DataFrame): Dataframe of postprocessed trial table
    """
    def __init__(
            self,
            df_trial: pd.DataFrame,
            sessionId: str,
            column_sessionId: str='sessionId',
            column_trialId: str='nTrial_raw',
            column_trial_start: Optional[str]=None,
            column_trial_end: Optional[str]=None,
            columns_alignment: Optional[List[str]]=None,
            columns_information_point: Optional[List[str]]=None,
            columns_information_broadcast: Optional[List[str]]=None,
            alignment_ignoreValues: Optional[List[Union[int, float]]]=[-1, 0],
            dict_column_drop_values: Optional[Dict[str, List]]=None,
            fillValue_sparse: Tuple[type, Any]=(float, 0.0),
            matlab_indexing: bool=True,
        ):

        self.df_trial_raw = df_trial
        self.sessionId = sessionId
        self.column_sessionId = column_sessionId
        self.column_trialId = column_trialId

        # Assert at least one of column_trial_start and column_trial_end is specified
        if column_trial_start is None and column_trial_end is None:
            raise ValueError('At least one of column_trial_start and column_trial_end must be specified.')
        elif column_trial_start is None and column_trial_end is not None:
            column_trial_start = column_trial_end
        elif column_trial_start is not None and column_trial_end is None:
            column_trial_end = column_trial_start
        self.column_trial_start = column_trial_start
        self.column_trial_end = column_trial_end

        self.columns_alignment = columns_alignment if columns_alignment is not None else list(df_trial.columns)
        self.columns_information_point = columns_information_point if columns_information_point is not None else []
        self.columns_information_broadcast = columns_information_broadcast if columns_information_broadcast is not None else []
        self.matlab_indexing = matlab_indexing 
        self.alignment_ignoreValues = alignment_ignoreValues if alignment_ignoreValues is not None else ([0] if matlab_indexing else [-1])
        self.dict_column_drop_values = dict_column_drop_values if dict_column_drop_values is not None else {}
        self.fillValue_sparse = fillValue_sparse

        self.df_trial = None
        self.df_trial_ids = None
        self.df_trial_links = None
        self.df_trial_sparse_ptr = None
        self.df_trial_sparse_data = None

    def preprocess(self):
        """
        Preprocess trial table data
        """
        self.df_trial = self.df_trial_raw.copy()

        self._drop_alignment_absent_rows()
        self._drop_rows_by_column_specified_values()
        self._alignment_columns_to_python_indexing() if self.matlab_indexing else None
        self._check_alignment_lowerBound(lowerBound=0)
        self._check_alignment_monotonic()
        
        self.df_trial[self.column_trialId] = np.arange(1, self.df_trial.shape[0]+1) if self.column_trialId not in self.df_trial.columns else self.df_trial[self.column_trialId]
        
        if self.column_trial_start is not None and self.column_trial_end is not None:
            self.column_trial_start = self.column_trial_start
        if self.column_trial_end is not None:
            self.column_trial_end = self.column_trial_end
        
        self.sparseDict_trialId = { # self.df_trial_ids = self.df_trial[lst_columns_trial_ids + [self.column_trial_start, self.column_trial_end]]
            'trial_start': {
                'row': self.df_trial[self.column_trial_start].values,
                'data': self.df_trial[self.column_trialId].values,
            },
            'trial_end': {
                'row': self.df_trial[self.column_trial_end].values,
                'data': self.df_trial[self.column_trialId].values,
            },
        }

        self.sparseDict_trialPoints = {
            'row': {column: self.df_trial[column].values for column in self.columns_alignment},
            'data': {column: self.df_trial[column].values for column in self.columns_information_point},
        }

        if self.sessionId is not None:
            assert self.column_sessionId not in self.df_trial.columns, f"sessionId column {self.column_sessionId} must not be in trial table if sessionId is specified"
            self.df_trial[self.column_sessionId] = self.sessionId
        else:
            assert self.column_sessionId in self.df_trial.columns, f"sessionId column {self.column_sessionId} must be in trial table if sessionId is not specified"
        lst_columns_trial_ids = [self.column_sessionId, self.column_trialId]

        self.broadcastDf_trialLinks = self.df_trial[lst_columns_trial_ids + self.columns_information_broadcast]
    
    def _drop_rows_by_column_specified_values(self):
        """
        Drop rows where any column contains a value specified in self.dict_column_drop_values
        """
        for column, lst_drop_values in self.dict_column_drop_values.items():
            self.df_trial = self.df_trial[~self.df_trial[column].isin(lst_drop_values)]

    def _drop_alignment_absent_rows(self):
        """
        Drop rows where any alignment column contains the value: self.alignment_absentValue
        """
        self.df_trial = self.df_trial[(~self.df_trial[self.columns_alignment].isin(self.alignment_ignoreValues)).all(axis=1)]
    
    def _alignment_columns_to_python_indexing(self):
        """
        Convert alignment columns to python indexing from matlab indexing
        """
        self.df_trial[self.columns_alignment] -= 1

    def _check_alignment_lowerBound(self, lowerBound: int=0):
        """
        Check alignment columns are above lower bound

        Args:
            lowerBound (int): Lower bound for alignment columns (Default is 0)
        """
        df_trial_valid = (self.df_trial[self.columns_alignment].isin(self.alignment_ignoreValues))|(self.df_trial[self.columns_alignment] >= lowerBound)
        assert np.all(df_trial_valid), "All alignment values must be >= `lowerBound` or in lst_ignoreValues."
    
    def _check_alignment_monotonic(self):
        """
        Check alignment columns are monotonically increasing
        """
        util.check_monotonically_increasing_df(
            self.df_trial[self.columns_alignment],
            lst_ignoreRowsWithValues=self.alignment_ignoreValues,
            equal_allowed=False,
        ) # Check that alignment values are monotonically increasing
    
class SignalPreprocessor():
    """
    Processor to handle signal data inlcuding:
    * Renaming columns

    JZ 2023

    Args:
        - df_signal (pd.DataFrame): Dataframe of signal data
        - columns_dense (Optional[List[str]]): List of column names that are dense
            and require saving all values. (e.g. signal traces, timestamps, session
            ids, trial ids, etc.). If ``None``, all columns are assumed to be dense.
            (Default is ``None``)
        - columns_sparse (Optional[List[str]]): List of column names that are sparse
            and require saving only a subset of values. (e.g. one-hot behavioral
            columns). If ``None``, all columns are assumed to not be sparse.
            (Default is ``None``)
        - fillValue_sparse (Tuple[type, Any]): Tuple of type and value with which to fill sparse
            information columns. (Default is ``(float, 0.0)``)

    Attributes:
        TODO
    """
    def __init__(
            self,
            df_signal: pd.DataFrame,
            sessionId: str,
            columns_dense: Optional[List[str]]=None,
            columns_sparse: Optional[List[str]]=None,
            fillValue_sparse: Tuple[type, Any]=(float, 0.0),
        ):
        self.df_signal_raw = df_signal
        self.sessionId = sessionId
        self.columns_dense = list(self.df_signal_raw.columns) if columns_dense is None else columns_dense
        self.columns_sparse = [] if columns_sparse is None else columns_sparse
        self.fillValue_sparse = fillValue_sparse

    def preprocess(self):
        """
        Preprocess signal data
        """
        self.df_signal = self.df_signal_raw.copy()
        self.denseDf_signal = self.df_signal[self.columns_dense]
        self.sparseDf_signal = self.df_signal[self.columns_sparse].astype(pd.SparseDtype(self.fillValue_sparse[0], fill_value=self.fillValue_sparse[1]))

class TrialSignalAligner():
    """
    Align trial table and signal data

    JZ 2023
    """
    def __init__(
            self,
            trialPreprocessor: Optional[TrialPreprocessor]=None,
            signalPreprocessor: Optional[SignalPreprocessor]=None,
            lst_trialsignalaligner: Optional[List['TrialSignalAligner']]=None,
            sessionId: Optional[str]=None,
        ):
        """
        Args:
            - trialPreprocessor (Optional[TrialPreprocessor]): TrialPreprocessor to preprocess trial table data
            - signalPreprocessor (Optional[SignalPreprocessor]): SignalPreprocessor to preprocess signal data
            - lst_trialsignalaligner (Optional[List['TrialSignalAligner']]): List of TrialSignalAligners
            - sessionId (str): Session id
        """
        assert (lst_trialsignalaligner is None) != (trialPreprocessor is None and signalPreprocessor is None), "Either lst_trialsignalaligner or trialPreprocessor and signalPreprocessor must be specified"
        if lst_trialsignalaligner is not None:
            self.trialPreprocessor = None
            self.signalPreprocessor = None
            self.from_list(lst_trialsignalaligner)
        else:
            self.trialPreprocessor = trialPreprocessor
            self.signalPreprocessor = signalPreprocessor
            self.df_aligned = None

        # Assert sessionId between trialPreprocessor and signalPreprocessor are the same and set sessionId if specified
        if self.trialPreprocessor is not None and self.signalPreprocessor is not None:
            assert self.trialPreprocessor.sessionId == self.signalPreprocessor.sessionId, "trialPreprocessor and signalPreprocessor must have the same sessionId"
            if sessionId is not None:
                assert sessionId == self.trialPreprocessor.sessionId, "sessionId must match trialPreprocessor sessionId"
                self.sessionId = sessionId
            else:
                self.sessionId = self.trialPreprocessor.sessionId

        self.sessionId = sessionId

    def from_list(self, lst_trialsignalaligner: List['TrialSignalAligner']):
        """
        Create TrialSignalAligner from list of TrialSignalAligners

        Args:
            lst_trialsignalaligner (List['TrialSignalAligner']): List of TrialSignalAligners
        """
        assert isinstance(lst_trialsignalaligner, list), "lst_trialsignalaligner must be a list"
        assert all([isinstance(trialsignalaligner, TrialSignalAligner) for trialsignalaligner in lst_trialsignalaligner]), "lst_trialsignalaligner must be a list of TrialSignalAligners"
        assert len(lst_trialsignalaligner) > 0, "lst_trialsignalaligner must have at least one TrialSignalAligner"
        self.df_aligned = pd.concat([trialsignalaligner.df_aligned for trialsignalaligner in lst_trialsignalaligner])

    def align(
            self,
        ):
        """
        Align trial table and signal data

        Args:
            TODO

        Returns:
            TODO
        """
        assert self.trialPreprocessor is not None and self.signalPreprocessor is not None, "trialPreprocessor and signalPreprocessor must be specified in __init__ to run align"
        self._check_alignment_upperBound(upperBound=self.signalPreprocessor.df_signal.shape[0], allow_equal=False)
        self._aligners_to_trialId() # 'trial_bounded_start', 'trial_bounded_end', 'trial_bounds_diff'
        self._aligners_to_onehots()
        self._alignerInfos_to_points()
        self._alignerInfos_to_broadcasts()
        self._check_alignedDfNames_notInSignal()
        self._combine_signalDf_alignedDf()
        self._drop_irrelevant_timestamps()
        self._add_timestamps() if self.str_timestamp_columns is not None else None
        self._assert_timestamps_monotonic_continuous() if self.str_timestamp_columns is not None else None
        
        # self.df_aligned = pd.concat([df_sessionId, df_trialIds, df_signal, df_timestamps, ], axis=1)
    
    def _check_alignment_upperBound(self, upperBound: int, allow_equal: bool=False):
        """
        Check alignment columns are below upper bound

        Args:
            upperBound (int): Upper bound for alignment columns
        """
        assert self.trialPreprocessor is not None and self.signalPreprocessor is not None, "trialPreprocessor and signalPreprocessor must be specified in __init__ to run align"
        if not allow_equal:
            assert np.all(self.trialPreprocessor.df_trial[self.trialPreprocessor.columns_alignment] < upperBound), "All alignment values must be < `upperBound` or in lst_ignoreValues."
        else:
            assert np.all(self.trialPreprocessor.df_trial[self.trialPreprocessor.columns_alignment] <= upperBound), "All alignment values must be <= `upperBound` or in lst_ignoreValues."

    def _aligners_to_trialId(self):
        """
        Align trial table and signal data to create trialId
        """
        assert self.trialPreprocessor is not None and self.signalPreprocessor is not None, "trialPreprocessor and signalPreprocessor must be specified in __init__ to run align"
        
        fill_value = np.nan

        sparr_trialId_setup, columnNames = util.sparse_dict_to_sparseArray(
            sparse_dict=self.trialPreprocessor.sparseDict_trialId,
            num_rows=self.signalPreprocessor.df_signal.shape[0],
        )
        self.denseDf_trialIds = pd.DataFrame(
            sparr_trialId_setup.todense(),
            columns=columnNames
        ).astype(float)
        self.denseDf_trialIds = self.denseDf_trialIds.replace(0.0, fill_value)
        
        # Create a new column ffill-ed from the combine first to create nTrial
        self.denseDf_trialIds['trial_bounded_start'] = self.denseDf_trialIds['trial_start'].ffill()
        assert np.all(self.denseDf_trialIds['trial_bounded_start'].diff().fillna(0) >= 0), "trial_bounded_start must be monotonically increasing"

        # Create a new column bfill-ed from the combine first to create nEndTrial
        self.denseDf_trialIds['trial_bounded_end'] = self.denseDf_trialIds['trial_end'].bfill()
        assert np.all(self.denseDf_trialIds['trial_bounded_end'].diff().fillna(0) >= 0), "trial_bounded_end must be monotonically increasing"

        # Create a new column that is the difference between nTrial and nEndTrial
        self.denseDf_trialIds['trial_bounds_diff'] = self.denseDf_trialIds['trial_bounded_end'] - self.denseDf_trialIds['trial_bounded_start']

        assert np.all(self.denseDf_trialIds['trial_bounds_diff'].dropna() >= 0), "trial_bounds_diff must be nan or >= 0 since every trial end must be preceded by a trial start"

        self.denseDf_trialIds = self.denseDf_trialIds[[
             'trial_bounded_start',
             'trial_bounded_end',
             'trial_bounds_diff'
        ]]

    def _aligners_to_onehots(self):
        """
        Align trial table to create onehots
        """
        assert self.trialPreprocessor is not None and self.signalPreprocessor is not None, "trialPreprocessor and signalPreprocessor must be specified in __init__ to run align"
        
        sparse_aligner_dict = {
            aligner_column: {
                'row': aligner_column_values
            } for aligner_column, aligner_column_values in self.trialPreprocessor.sparseDict_trialPoints['row'].items()
        }

        sparr_aligners_setup, columnNames = util.sparse_dict_to_sparseArray(
            sparse_dict=sparse_aligner_dict,
            num_rows=self.signalPreprocessor.df_signal.shape[0],
        )

        self.sparseDf_onehots = pd.DataFrame.sparse.from_spmatrix(
            sparr_aligners_setup,
            columns=columnNames
        ).astype(float)

    def _alignerInfos_to_points(self):
        """
        Align trial table to create points
        """
        assert self.trialPreprocessor is not None and self.signalPreprocessor is not None, "trialPreprocessor and signalPreprocessor must be specified in __init__ to run align"
        
        # Set self.sparseDf_points to None, print a message, and return if the lengths of either self.trialPreprocessor.sparseDict_trialPoints['row'] or self.trialPreprocessor.sparseDict_trialPoints['data'] is zero
        if len(self.trialPreprocessor.sparseDict_trialPoints['row']) == 0 or len(self.trialPreprocessor.sparseDict_trialPoints['data']) == 0:
            self.sparseDf_points = pd.DataFrame.sparse(pd.DataFrame(index=self.signalPreprocessor.df_signal.index))
            print("self.trialPreprocessor.sparseDict_trialPoints['row'] or self.trialPreprocessor.sparseDict_trialPoints['data'] is empty. Setting self.sparseDf_points to empty dataframe.")
            return

        sparse_alignerInfo_dict = {}
        for aligner_column, aligner_column_values in self.trialPreprocessor.sparseDict_trialPoints['row'].items():
            for info_column, info_column_values in self.trialPreprocessor.sparseDict_trialPoints['data'].items():
                sparse_alignerInfo_dict[(aligner_column, info_column)] = {
                    'row': aligner_column_values,
                    'data': info_column_values
                }

        sparr_alignerInfo_setup, columnNames = util.sparse_dict_to_sparseArray(
            sparse_dict=sparse_alignerInfo_dict,
            num_rows=self.signalPreprocessor.df_signal.shape[0],
        )

        self.sparseDf_points = pd.DataFrame.sparse.from_spmatrix(
            sparr_alignerInfo_setup,
            columns=columnNames
        ).astype(float)

        print()

    # def _alignerInfos_to_broadcasts(self):
    #     broadcastDf_trialLinks

        
    ############################################################################################################
    ############################################################################################################
    #         keep_row_columnName: Optional[str]=None,

    #     - keep_row_column (Optional[str]): Column of True/False values to
    #       indicate whether or not keep rows (True = keep, False = drop)
    #     self.keep_row_columnName = keep_row_columnName
    #     self._drop_rows_non_keep_row_column() if self.keep_row_columnName is not None else None

    # def _drop_rows_non_keep_row_column(self):
    #     """
    #     Drop rows of signal data that are False in keep_row_column
    #     """
    #     assert self.keep_row_columnName is not None, "keep_row_columnName must be specified"
    #     assert self.keep_row_columnName in self.df_signal.columns, f"keep_row_columnName: {self.keep_row_columnName} not in df_signal.columns: {self.df_signal.columns}"
    #     self.df_signal = self.df_signal[self.df_signal[self.keep_row_columnName]]
    ############################################################################################################
    ############################################################################################################


if __name__ == '__main__':
    dir_data = Path('/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/data/old-data-version/raw-new/Figure_1_2')
    dir_output = Path('/Users/josh/Desktop/example_output_folder')

    dict_inputdata = {'session_id': 'WT63_11102021',
    'filepath_signal': dir_data / Path('GLM_SIGNALS_WT63_11102021.txt'),
    'filepath_trial': dir_data / Path('GLM_TABLE_WT63_11102021.txt'),
    'bool_trialTable_matlab_indexed': True,
    'columnName_trialTable_trialId': None,
    'columnRenames_signal': {'Ch1': 'gDA', 'Ch5': 'gACH'},
    'columnRenames_trial': {
                'photometryCenterInIndex': 'CI',
                'photometryCenterOutIndex': 'CO',
                'photometrySideInIndex': 'SI',
                'photometrySideOutIndex': 'SO',
                'hasAllPhotometryData': 'hasAllData',
                'wasRewarded': 'rew',
                'word': 'wd',
        }
    }

    columnName_alignment_trial_start = 'CI'
    columnName_alignment_trial_end = 'SO'

    # Note: Alignment values of 0 for Matlab-indexed trial tables will be treated as "no-data" values
    # and and -1 for Python-indexed trial tables. Matlab-indexed trial tables should only have values
    # >= 0 in and >= -1 in Python.
    lst_columnNames_trialTable_alignment = [
        'CI',
        'CO',
        'SI',
        'SO',
    ]

    lst_columnNames_trialTable_information = [
        # 'nTrial_raw', 
        'hasAllData',
        'rew', 'wd',
    ]
    
    sessionId = 'test1'

    # Test trial table preprocessing
    trial = TrialPreprocessor(
        pd.read_csv(
            dict_inputdata['filepath_trial']
        ).rename(dict_inputdata['columnRenames_trial'], axis=1),
        sessionId = sessionId,
        column_sessionId = 'sessionId',
        column_trialId = 'nTrial_raw',
        column_trial_start = columnName_alignment_trial_start,
        column_trial_end = columnName_alignment_trial_end,
        columns_alignment = lst_columnNames_trialTable_alignment,
        columns_information_point = ['wd'],
        columns_information_broadcast = lst_columnNames_trialTable_information,
        matlab_indexing = True,
        alignment_ignoreValues = [-1, 0],
        dict_column_drop_values = {},
    )
    trial.preprocess();

    signal = SignalPreprocessor(
        pd.read_csv(
            dict_inputdata['filepath_signal']
        ).rename(dict_inputdata['columnRenames_signal'], axis=1),
        columns_sparse=[],
        columns_dense=[],
        sessionId = sessionId,
    )
    signal.preprocess();

    trialSignalAligned = TrialSignalAligner(trial, signal, sessionId=sessionId)
    trialSignalAligned.align()
    print()



# if __name__ == "__main__":
#     dir_data = Path('/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/data/old-data-version/raw-new/Figure_1_2')
#     dir_output = Path('/Users/josh/Desktop/example_output_folder')

#     lst_dict_inputdata = [
#         {'session_id': 'WT63_11082021',
#         'filepath_signal': dir_data / Path('GLM_SIGNALS_WT63_11082021.txt'),
#         'filepath_trial': dir_data / Path('GLM_TABLE_WT63_11082021.txt'),
#         'bool_trialTable_matlab_indexed': True,
#         'columnName_trialTable_trialId': None,
#         'columnRenames_signal': {'Ch1': 'gDA', 'Ch5': 'gACH'},
#         'columnRenames_trial': None},
#     ]

#     dir_output.mkdir(parents=True, exist_ok=True)

#     columnName_alignment_trial_start = 'photometryCenterInIndex'
#     columnName_alignment_trial_end = 'photometrySideOutIndex'

#     # Note: Alignment values of 0 for Matlab-indexed trial tables will be treated as "no-data" values
#     # and and -1 for Python-indexed trial tables. Matlab-indexed trial tables should only have values
#     # >= 0 in and >= -1 in Python.
#     lst_strColumns_alignment = [
#         'photometryCenterInIndex',
#         'photometryCenterOutIndex',
#         'photometrySideInIndex',
#         'photometrySideOutIndex',
#     ]

#     lst_strColumns_information = [
#         'nTrial_raw', 'hasAllPhotometryData',
#         'wasRewarded', 'word',
#     ]

#     bool_drop_zeroAlignments = True

#     lst_trialSignalAligned = []
#     for dict_inputdata in lst_dict_inputdata:
#         # Load data
#         trial = TrialPreprocessor(dict_inputdata['filepath_trial'])
#         signal = SignalPreprocessor(dict_inputdata['filepath_signal'])

#         # Preprocess trial table
#         trial.preprocess();
#         signal.preprocess();

#         # Trial / signal alignment
#         trialSignalAligned = TrialSignalAligner(trial, signal)
#         trialSignalAligned.align();
#         trialSignalAligned.trialstamp();
#         trialSignalAligned.timestamp();

#         # Aggregate
#         lst_trialSignalAligned.add(trialSignalAligned);

#     trialSignalAligned_agg.combine();

#     # Generate prediction dataframe X, prediction dataframe y
#     predictors = ['predictor_1', 'predictor_2']
#     response = 'y'
#     trialSignalAligned_agg.generate_Xy(predictors, response);

#     # Unroll specified X columns into onehot representations
#     trialSignalAligned_agg.unroll_X_columns(['predictor_1', 'predictor_2']);

#     # Timeshift X columns
#     trialSignalAligned_agg.timeshift_X_columns(['predictor_1', 'predictor_2'], shift_amt=1);

#     # Split train/validation/test sets
#     trialSignalAligned_agg.split_train_validation_test();

#     # Fit GLM
#     glm = GLM(trialSignalAligned_agg);
#     glm.fit_GLM();
#     glm.generate_GLM_summary();
#     glm.plot_GLM_summary();

#     # Generate predictions for train/validation/test sets. Evaluate predictions on train/validation/test sets.
#     glm.generate_predictions();
#     glm.evaluate_predictions();
#     glm.generate_prediction_plots();

#     # Save preprocessing parameters
#     trial.save_preprocessing_info(dir_output / Path('trial_preprocessing_info.json'));
#     signal.save_preprocessing_info(dir_output / Path('signal_preprocessing_info.json'));

#     # Save alignment parameters
#     trialSignalAligned.save_alignment_info(dir_output / Path('alignment_info.json'));

#     # Save aggregation parameters
#     trialSignalAligned_agg.save_aggregation_info(dir_output / Path('aggregation_info.json'));

#     # Save GLM parameters
#     glm.save_GLM_info(dir_output / Path('glm_info.json'));
