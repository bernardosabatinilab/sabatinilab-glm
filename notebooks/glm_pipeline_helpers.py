from typing import DefaultDict, List, TypeVar, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import matplotlib.pyplot as plt

class TrialPreprocessor():
    """
    Preprocess trial table data

    JZ 2023

    Args:
        TODO

    Attributes:
        TODO
    """
    def __init__(
            self,
            filepath_trial: Path,
            sep: Optional[str] = None,
            columnName_trialId: Optional[str]=None,
            columnName_alignment_trial_start: Optional[str]=None,
            columnName_alignment_trial_end: Optional[str]=None,
            lst_strColumns_alignment: Optional[List[str]]=None,
            lst_strColumns_information_single: Optional[List[str]]=None,
            lst_strColumns_information_broadcast: Optional[List[str]]=None,
            bool_trialTable_matlab_indexed: Optional[bool]=True,
            alignment_dummyValue: Optional[int]=None,
        ):
        self.filepath_trial = filepath_trial
        self.df_trial = pd.read_csv(filepath_trial) if not sep else pd.read_csv(filepath_trial, sep=sep)
        self.columnName_trialId = columnName_trialId
        self.columnName_alignment_trial_start = columnName_alignment_trial_start
        self.columnName_alignment_trial_end = columnName_alignment_trial_end
        self.lst_strColumns_alignment = lst_strColumns_alignment
        self.lst_strColumns_information_single = lst_strColumns_information_single
        self.lst_strColumns_information_broadcast = lst_strColumns_information_broadcast
        self.bool_trialTable_matlab_indexed = bool_trialTable_matlab_indexed

        if alignment_dummyValue is not None:
            self.alignment_dummyValue = alignment_dummyValue
        else:
            self.alignment_dummyValue = 0 if self.bool_trialTable_matlab_indexed else -1

    def preprocess(self):
        """
        Preprocess trial table data

        Args:
            TODO

        Returns:
            TODO
        """
        pass
        
class SignalPreprocessor():
    """
    Preprocess signal data

    JZ 2023

    Args:
        TODO
    
    Attributes:
        TODO
    """
    def __init__(
            self,
            filepath_signal: Path,
            sep: Optional[str] = None,
            columnRenames_signal: Optional[dict]=None,
        ):
        self.filepath_signal = filepath_signal
        self.df_signal = pd.read_csv(filepath_signal) if not sep else pd.read_csv(filepath_signal, sep=sep)
        self.columnRenames_signal = columnRenames_signal

    def preprocess(self):
        """
        Preprocess signal data

        Args:
            TODO

        Returns:
            TODO
        """
        pass

class TrialSignalAligner():
    """
    Align trial table and signal data

    JZ 2023

    Args:
        TODO

    Attributes:
        TODO
    """
    def __init__(
            self,
            trialPreprocessor: TrialPreprocessor,
            signalPreprocessor: SignalPreprocessor,
        ):
        self.trialPreprocessor = trialPreprocessor
        self.signalPreprocessor = signalPreprocessor
        self.df_alignedSignal = self.signalPreprocessor.df_signal.copy()

    def align(self):
        """
        Align trial table and signal data

        Args:
            TODO

        Returns:
            TODO
        """
        df_trial = self.trialPreprocessor.df_trial
        df_signal = self.signalPreprocessor.df_signal
        columnName_trialId = self.trialPreprocessor.columnName_trialId

        if alignment_absent is not None:
            self.alignment_absent = alignment_absent
        else:
            self.alignment_absent = alignment_dummyValue

        # Add trial number column to trial table
        df_trial = _add_column_nTrial_raw(df_trial, columnName_trialId)

        # Convert trial table alignment information to signal columns
        df_signal_aligned = alignmentInfoTrialPairs_to_signalColumns(
            df_trial,
            lst_strColumns_alignment,
            df_signal.index,
            fillValues_absent=self.alignment_absent,
            fillValues_present_noInfo=1.0,
        )

        # Convert trial table information to signal columns
        df_signal_aligned = informationTrialPairs_to_signalColumns(
            df_trial,
            lst_strColumns_information_single,
            df_signal_aligned,
            fillValues_absent=self.alignment_absent,
            fillValues_present_noInfo=1.0,
        )

        # Convert trial table broadcast information to signal columns
        df_signal_aligned = informationBroadcastTrialPairs_to_signalColumns(
            df_trial,
            lst_strColumns_information_broadcast,
            df_signal_aligned,
            fillValues_absent=self.alignment_absent,
            fillValues_present_noInfo=1.0,
        )

        # Remove trial number column from trial table
        df_trial = df_trial.drop(columns=[columnName_trialId])

        # Remove alignment information columns from trial table
        df_trial = df_trial.drop(columns=lst_strColumns_alignment)

        # Remove information columns from trial table
        df_trial = df_trial.drop(columns=lst_strColumns_information_single)

        # Remove broadcast information columns from trial table
        df_trial = df_trial.drop(columns=lst_strColumns_information)


def alignmentInfoTrialPairs_to_signalColumns(
        df_trial: pd.DataFrame,
        lst_columnNamePairs_trial_alignInfo: List[Tuple[str, str]],
        df_signal_index: pd.Index,
        fillValues_absent: float=np.nan,
        fillValues_present_noInfo: float=1.0
    ):
    """
    Convert a dataframe of trial table alignment information to a dataframe of signal columns.
    (Each row in the output dataframe corresponds to a row in df_signal dataframe.)
    (Each column in the output dataframe corresponds to a column in lst_columnNamePairs_trial_alignInfo.)
    (The output dataframe will have the same number of rows as df_signal_index and the same number of columns as lst_columnNamePairs_trial_alignInfo.)

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
            fillValues_absent = fillValues_absent
            fillValues_present = fillValues_present_noInfo
        else:
            output_column_name = f"{str_columnPair_alignment}={str_columnPair_information}"
            fillValues_absent = fillValues_absent
            fillValues_present = df_trial[str_columnPair_information]
        

        tmp = np.full((len(df_signal_index)), fillValues_absent).astype(object)
        tmp[df_trial[str_columnPair_alignment].values] = fillValues_present
        lst_tmp.append(tmp)
        lst_output_column_names.append(output_column_name)

    return pd.DataFrame(np.stack(lst_tmp, axis=1), columns=lst_output_column_names, index=df_signal_index)

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

if __name__ == "__main__":
    dir_data = Path('/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/data/old-data-version/raw-new/Figure_1_2')
    dir_output = Path('/Users/josh/Desktop/example_output_folder')

    lst_dict_inputdata = [
        {'session_id': 'WT63_11082021',
        'filepath_signal': dir_data / Path('GLM_SIGNALS_WT63_11082021.txt'),
        'filepath_trial': dir_data / Path('GLM_TABLE_WT63_11082021.txt'),
        'bool_trialTable_matlab_indexed': True,
        'columnName_trialTable_trialId': None,
        'columnRenames_signal': {'Ch1': 'gDA', 'Ch5': 'gACH'},
        'columnRenames_trial': None},
    ]

    dir_output.mkdir(parents=True, exist_ok=True)

    columnName_alignment_trial_start = 'photometryCenterInIndex'
    columnName_alignment_trial_end = 'photometrySideOutIndex'

    # Note: Alignment values of 0 for Matlab-indexed trial tables will be treated as "no-data" values
    # and and -1 for Python-indexed trial tables. Matlab-indexed trial tables should only have values
    # >= 0 in and >= -1 in Python.
    lst_strColumns_alignment = [
        'photometryCenterInIndex',
        'photometryCenterOutIndex',
        'photometrySideInIndex',
        'photometrySideOutIndex',
    ]

    lst_strColumns_information = [
        'nTrial_raw', 'hasAllPhotometryData',
        'wasRewarded', 'word',
    ]

    bool_drop_zeroAlignments = True

    trialSignalAligned_agg = TrialSignalAlignerAggregator()

    for dict_inputdata in lst_dict_inputdata:
        # Load data
        trial = TrialPreprocessor(dict_inputdata['filepath_trial'])
        signal = SignalPreprocessor(dict_inputdata['filepath_signal'])

        # Preprocess trial table
        trial.preprocess();
        signal.preprocess();

        # Trial / signal alignment
        trialSignalAligned = TrialSignalAligner(trial, signal)
        trialSignalAligned.align();
        trialSignalAligned.trialstamp();
        trialSignalAligned.timestamp();

        # Aggregate
        trialSignalAligned_agg.add(trialSignalAligned);

    trialSignalAligned_agg.combine();

    # Generate prediction dataframe X, prediction dataframe y
    predictors = ['predictor_1', 'predictor_2']
    response = 'y'
    trialSignalAligned_agg.generate_Xy(predictors, response);

    # Unroll specified X columns into onehot representations
    trialSignalAligned_agg.unroll_X_columns(['predictor_1', 'predictor_2']);

    # Timeshift X columns
    trialSignalAligned_agg.timeshift_X_columns(['predictor_1', 'predictor_2'], shift_amt=1);

    # Split train/validation/test sets
    trialSignalAligned_agg.split_train_validation_test();

    # Fit GLM
    glm = GLM(trialSignalAligned_agg);
    glm.fit_GLM();
    glm.generate_GLM_summary();
    glm.plot_GLM_summary();

    # Generate predictions for train/validation/test sets. Evaluate predictions on train/validation/test sets.
    glm.generate_predictions();
    glm.evaluate_predictions();
    glm.generate_prediction_plots();

    # Save preprocessing parameters
    trial.save_preprocessing_info(dir_output / Path('trial_preprocessing_info.json'));
    signal.save_preprocessing_info(dir_output / Path('signal_preprocessing_info.json'));

    # Save alignment parameters
    trialSignalAligned.save_alignment_info(dir_output / Path('alignment_info.json'));

    # Save aggregation parameters
    trialSignalAligned_agg.save_aggregation_info(dir_output / Path('aggregation_info.json'));

    # Save GLM parameters
    glm.save_GLM_info(dir_output / Path('glm_info.json'));



# # End result should function the same as...
# if __name__ == '__main__':
#     columnName_alignment_trial_start = 'photometryCenterInIndex'
#     columnName_alignment_trial_end = 'photometrySideOutIndex'

#     # Note: Alignment values of 0 for Matlab-indexed trial tables will be treated as "no-data" values
#     # and and -1 for Python-indexed trial tables. Matlab-indexed trial tables should only have values
#     # >= 0 in and >= -1 in Python.
#     lst_columnNames_trialTable_alignment = [
#         'photometryCenterInIndex',
#         'photometryCenterOutIndex',
#         'photometrySideInIndex',
#         'photometrySideOutIndex',
#     ]

#     lst_columnNames_trialTable_information = [
#         'nTrial_raw', 'hasAllPhotometryData',
#         'wasRewarded', 'word',
#     ]

#     lst_columnNamePairs_trial_alignInfo = list(itertools.product(
#         lst_columnNames_trialTable_alignment,
#         lst_columnNames_trialTable_information,
#     ))

#     lst_columnNamePairs_trial_alignInfo = (
#         [(columnName_alignment, columnName_alignment) for columnName_alignment in lst_columnNames_trialTable_alignment] +
#         lst_columnNamePairs_trial_alignInfo
#     )

#     bool_drop_zeroAlignments = True
#     lst_df_signal_all = []

#     for dict_inputdata in lst_dict_inputdata:
#         columnName_trialTable_trialId = dict_inputdata.get('columnName_trialTable_trialId', None)

#         # Load data
#         df_trial = pd.read_csv(dict_inputdata['filepath_trial'])
#         df_signal = pd.read_csv(dict_inputdata['filepath_signal'])
#         df_signal = df_signal.rename(dict_inputdata['columnRenames_signal'], axis=1)

#         check_monotonically_increasing_df(df_trial[lst_columnNames_trialTable_alignment], bool_ignoreRowsWithZeros=True, bool_ignoreRowsWithNans=True, equal_allowed=False) # Check that alignment values are monotonically increasing

#         df_trial = df_trial[df_trial[lst_columnNames_trialTable_alignment].values.min(axis=1) > 0].copy() if bool_drop_zeroAlignments else df_trial.copy() # Drop trials with no data
#         columnName_trialTable_trialId, df_trial = ('nTrial_raw', add_column_nTrial_raw(df_trial, 'nTrial_raw')) if columnName_trialTable_trialId is None else (columnName_trialTable_trialId, df_trial.copy()) # Add trial ID column if not already present
        
#         if not dict_inputdata['bool_trialTable_matlab_indexed']:
#             df_trial = df_trial[lst_columnNames_trialTable_alignment] + 1 # Convert to Matlab-indexed trial table
#         assert df_trial[lst_columnNames_trialTable_alignment].values.min() >= 0, "All alignment values must be >= 0 for Matlab-indexed trial tables."

#         df_signal = pd.concat([pd.DataFrame(pd.Series(0, index=df_signal.columns)).T, df_signal], axis=0).reset_index(drop=True) # Add a row of zeros to the top of the signal dataframe to absorb "no-data" values
#         df_alignmentInfoPairs = alignmentInfoTrialPairs_to_signalColumns(df_trial, lst_columnNamePairs_trial_alignInfo, df_signal.index)

#         assert len(set(df_alignmentInfoPairs.columns).intersection(set(df_signal.columns))) == 0, "Alignment info dataframe must not have any columns that appear in the signal dataframe."

#         df_signal = pd.concat([df_signal, df_alignmentInfoPairs], axis=1)

#         assert len(set([columnName_trialTable_trialId, 'nTrial', 'nEndTrial']).intersection(set(df_signal.columns))) == 0, f"Signal dataframe must not have any columns named 'nTrial', 'nEndTrial', or User Specified: '{columnName_trialTable_trialId}' name."
#         df_signal[columnName_trialTable_trialId] = df_signal[f'{columnName_alignment_trial_start}={columnName_trialTable_trialId}'].combine_first(df_signal[f'{columnName_alignment_trial_end}={columnName_trialTable_trialId}'])
#         df_signal.insert(0, 'session_id', dict_inputdata['session_id'])
#         df_signal.insert(1, 'nTrial', df_signal[columnName_trialTable_trialId].ffill())
#         df_signal.insert(2, 'nEndTrial', df_signal[columnName_trialTable_trialId].bfill())
#         check_monotonically_increasing_df(df_signal[['nTrial']], bool_ignoreRowsWithZeros=False, bool_ignoreRowsWithNans=True, equal_allowed=True)
#         check_monotonically_increasing_df(df_signal[['nEndTrial']], bool_ignoreRowsWithZeros=False, bool_ignoreRowsWithNans=True, equal_allowed=True)

#         df_signal = df_signal.drop(columnName_trialTable_trialId, axis=1).copy()
#         df_signal = df_signal.drop(0, axis=0).copy()
#         df_signal.insert(3, 'timestamp', np.arange(df_signal.shape[0])) # / dict_inputdata['sampling_rate']
#         lst_df_signal_all.append(df_signal)

#     df_signal_all = pd.concat(lst_df_signal_all, axis=0).reset_index(drop=True)
#     df_signal_all.to_csv(dir_output / 'df_signal_all.csv', index=False)