from typing import DefaultDict, List, TypeVar, Optional, Union, List
import numpy as np
import pandas as pd
import scipy.signal
from numba import njit, jit, prange

import sys
import threading
from sklearn.model_selection import GroupShuffleSplit

# import caiman 
# If this causes an error, navigate to backend/lib/CaImAn and run:
#     pip install -r requirements.txt
#     pip install .
### Suite 2p's regularization / deconvolution


# TODO: Write testcases & check validity
# TODO: Include train/test split - by 2 min segmentation
# TODO: Switch to suite2p's convolution
# TODO: Numba impolementations

def timeshift(X, shift_inx=[], shift_amt=1, keep_non_inx=False, dct=None, fill_value=np.nan):
    """
    Shift columns in shift_inx up or down by shift_amt (down if shift_amt > 0, up if shift_amt < 0)

    JZ 2021
    
    Args:
        X : np.ndarray or pd.DataFrame
            Array of all variables (columns should be features, rows should be timesteps)
        shift_inx : list(int)
            Column indices to shift forward/backward
        shift_amt : int
            Amount by which to shift the columns in question up or down (down if shift_amt > 0, up if shift_amt < 0)
        keep_non_inx : bool
            If True, data from all columns (shifted or not) will be returned from the function.
            If False, only shifted columns are returned.
        dct : dict
            Dictionary for in-place timeshift updates
        fill_value : np.float
            Value to be left in place of shifted data
    
    Returns: DataFrame or NDArray including the relevant timeshift
    """
    
    npX = get_numpy_version(X)
    # Use all columns for shifting if none specified
    shift_inx = range(npX.shape[1]) if len(shift_inx) == 0 else shift_inx
    X_to_shift = npX[:, shift_inx]
    shifted_X = shift(X_to_shift, shift_amt, fill_value=fill_value)
    return_setup = shifted_cols_to_original_type(X, shifted_X, shift_inx, keep_non_inx)
    
    if dct is not None:
        dct[shift_amt] = return_setup
    return return_setup

def timeshift_multiple(X, shift_inx=[], shift_amt_list=[-1,0,1], unshifted_keep_all=True, fill_value=np.nan):
    """
    Collect multiple up/down shifts of columns as columns in the returned array

    JZ 2021
    
    Args:
        X : np.ndarray (preferably contiguous array)
            Array of all variables (columns should be features, rows should be timesteps)
        shift_inx : list(int)
            Column indices to shift forward/backward
        shift_amt_list : list(int)
            List of amounts by which to shift forward the columns in question (backward where list elements are < 0)
        unshifted_keep_all : bool
            Whether or not to keep all unshifted columns in the returned array
        fill_value : np.float
            Value to be left in place of shifted data
    
    Returns: DataFrame or NDArray with the relevant timeshifts
    """

    # print(shift_inx)

    shifted_dict = {}
    threads = []
    for i,shift_amt in enumerate(shift_amt_list):
        threads.append(threading.Thread(target=timeshift, args=(X,), kwargs={'shift_inx':shift_inx,
                                                                 'shift_amt':shift_amt,
                                                                 'keep_non_inx':(shift_amt == 0 and unshifted_keep_all),
                                                                 'dct':shifted_dict,
                                                                 'fill_value':fill_value
                                                                 }))
        threads[-1].start()

        if i % 20 == 19:
            for thread in threads:
                thread.join()
            threads = []

    for thread in threads:
        thread.join()

    # print(shifted_dict.keys())
    
    shifted_list = [shifted_dict[_] for _ in shift_amt_list]
    return concat_all_shifts(X, shift_amt_list, shifted_list)

def zscore(X):
    """
    Convert X values to z-scores along the 0th axis

    JZ 2021
    
    Args:
        X : np.ndarray or pd.DataFrame
            Array of variables to zscore
    
    Returns: Z-scored values of X along dimension 0
    """
    return (X - X.mean(axis=0))/X.std(axis=0)


def diff(X, diff_inx=[], n=1, axis=0, append_to_base=False, fill_value=np.nan, **kwargs):
    """
    Return the differential between each timestep and the previous timestep, n times.
    
    Documentation adopted from numpy diff.

    JZ 2021
    
    Args:
        X : np.ndarray (preferably contiguous array)
            Array of all variables (columns should be features, rows should be timesteps)
        diff_inx : list
            Column indices that should be diffed
        n : int, optional
            The number of times values are differenced. If zero, the input
            is returned as-is.
        axis : int, optional
            The axis along which the difference is taken, default is the
            first axis.
        append_to_base : bool
            Whether or not to keep the undifferenced columns in the returned array
        fill_value : float
            Value that should fill in the columns that do not have a prior column with
            which to difference
        **kwargs : prepend, append : array_like, optional
            Values to prepend or append to `a` along axis prior to
            performing the difference.  Scalar values are expanded to
            arrays with length 1 in the direction of axis and the shape
            of the input array in along all other axes.  Otherwise the
            dimension and shape must match `a` except along axis.

            Other keyword arguments for np diff.
    
    Returns: DataFrame or NDArray with the relevant columns differenced
    """

    typ = type(X)
    typ = pd.DataFrame if typ == pd.Series and append_to_base else typ

    if type(X) == pd.Series:
        X = pd.DataFrame(X)

    diff_inx = diff_inx if diff_inx else list(range(X.shape[1]))

    if type(X) == pd.DataFrame:
        column_names = [_ + '_diff' for _ in X.columns[diff_inx]]
        if append_to_base:
            column_names = list(X.columns) + column_names
        X_val = X.values
    else:
        X_val = X
    
    if len(X.shape) == 1:
        X_val = X_val.reshape((-1,1))
    
    ret = np.diff(X_val[:,diff_inx], n=n, axis=axis, **kwargs)

    if append_to_base:
        ret = np.concatenate([np.ones((n, ret.shape[1])) * fill_value, ret], axis=0)
        ret = np.concatenate([X_val, ret], axis=-1)
        if type(X) == pd.DataFrame:
            index = X.index
    elif type(X) == pd.DataFrame:
        index = X.index[1:]

    if type(X) == pd.DataFrame:
        ret = pd.DataFrame(ret, columns=column_names, index=index)
    if typ == pd.Series:
        ret = ret.iloc[:, 0]
    
    return ret

def get_column_nums(df, column_names=[]): #
    """
    Returns a list of the column numbers associated with each column name.

    JZ 2021
    
    Args:
        df : pd.DataFrame
            DataFrame of which to find column name index numbers
        column_names : array_like, optional
            Names of the columns of which to find column numbers
    
    Returns: list of the column indices associated with the names of columns in questions
    """
    ret = [df.columns.get_loc(_) for _ in column_names]
    if len([_ for _ in ret if type(_) == np.ndarray]):
        raise ValueError('Duplicate column found in X column names.')
    return ret








def bucket_ids_by_timeframe(total_timesteps, timesteps_per_bucket=20):
    """
    Create a numpy array of ids to associate each timestep with its nearest timesteps_per_bucket timesteps.
    
    Args:
        total_timesteps : int
            Number of timesteps in the overall dataset
        timesteps_per_bucket : int
            Number of timesteps that should fall within each time bucket

    Returns: numpy array of bucket ids of length 'total_timesteps'
    """
    # Step 1: Create time buckets of 1000 entries each (actual value will vary based on
    # sampling rate and amount of time desired for bucketing)
    num_buckets = total_timesteps // timesteps_per_bucket
    bucket_ids = np.arange(total_timesteps) // num_buckets
    return bucket_ids

def cv_idx_from_bucket_ids(bucket_ids, X, y=None, num_folds=None, test_size=None):
    '''
    Generate a GroupShuffleSplit on the provided bucket ids. (i.e. create a list of
    splits to use in different stages of cross-validation).

    Args:
        bucket_ids : np.ndarray
            Array of grouping ids to use in generating the GroupShuffleSplit
        X : pd.DataFrame
            Prediction DataFrame from which to bucket
        y : pd.Series
            Response Series
        num_folds : int
            Number of Cross Validation segmentations that should be used for GroupShuffleSplit fold Cross Validation
        test_size : float
            Percentage of datapoints to use in each GroupShuffleSplit fold for validation (Defaults to 1/num_folds if None)
    '''
    # Step 2: Create index sets for each of the K-folds based on prior groupings
    if num_folds is None:
        # Defaults to Leave One Out Cross-Validation
        num_folds = bucket_ids.max() + 1
    
    if test_size is None:
        # Defaults to the number of validation datapoints equal to a single fold
        test_size = 1/num_folds
    
    splitter = GroupShuffleSplit(n_splits=num_folds, test_size=test_size)
    cv_idx = list(splitter.split(X, y, bucket_ids))
    return cv_idx



######### HELPER FUNCTIONS #########
######### HELPER FUNCTIONS #########
######### HELPER FUNCTIONS #########
######### HELPER FUNCTIONS #########
######### HELPER FUNCTIONS #########
######### HELPER FUNCTIONS #########
######### HELPER FUNCTIONS #########
######### HELPER FUNCTIONS #########
######### HELPER FUNCTIONS #########
######### HELPER FUNCTIONS #########


def get_numpy_version(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Return a numpy array of values regardless of input type X (np.ndarray or pd.DataFrame)
    
    JZ 2021
    
    Args:
        X: Dataset to convert to numpy array
    
    Returns:
        npX: Numpy array version of original dataset
    """
    if type(X) == pd.DataFrame:
        npX = X.values
    else:
        npX = X
    return npX

def shift(setup_array: np.ndarray, shift_amt: int, fill_value: Optional[float] = np.nan) -> np.ndarray:
    """
    Shift all of setup_array up or down by shift_amt (if > 0: shift down, if < 0: shift up)
    
    JZ 2021
    
    Args:
        setup_array: Array to be shifted up or down
        shift_amt: Amount to shift data up or down (> 0 = shift down, < 0 = shift up)
        fill_value: Optional; Value to be left in place of shifted data
    
    Returns:
        shifted_X : Post-shift version of setup_array
    """
    blanks = np.ones((np.abs(shift_amt), setup_array.shape[1])) * fill_value
    if shift_amt > 0:
        shifted_X = concat_start_crop_end(blanks, setup_array)
    elif shift_amt < 0:
        shifted_X = concat_end_crop_start(blanks, setup_array)
    else:
        shifted_X = setup_array
    return shifted_X

def concat_start_crop_end(blanks: np.ndarray, X_to_shift: np.ndarray):
    """
    Concatenates blanks to the top of X_to_shift and crops the bottom such that
    dimensions of output == dimensions of input.
    
    JZ 2021
    
    Parameters:
        blanks: Values to be concatenated to the top of X_to_shift
        X_to_shift: Data to be concatenated and cropped at the bottom of the returned values
    
    Returns:
        shifted_X: Values of X_to_shift after concatenation and cropping of data
    """
    shift_amt = blanks.shape[0]
    shifted_X = np.concatenate((blanks, X_to_shift), axis=0)
    shifted_X = shifted_X[:-shift_amt, :]
    return shifted_X

def concat_end_crop_start(blanks: np.ndarray, X_to_shift: np.ndarray):
    """
    Concatenates blanks to the bottom of X_to_shift and crops the top such that
    dimensions of output == dimensions of input.
    
    JZ 2021
    
    Parameters:
        blanks: Values to be concatenated to the bottom of X_to_shift
        X_to_shift: Data to be concatenated and cropped at the top of the returned values
    
    Returns:
        shifted_X: Values of X_to_shift after concatenation and cropping of data
    """
    shift_amt = blanks.shape[0]
    shifted_X = np.concatenate((X_to_shift, blanks), axis=0)
    shifted_X = shifted_X[shift_amt:, :]
    return shifted_X

def shifted_cols_to_original_type(X: Union[np.ndarray, pd.DataFrame],
                                  shifted_X: np.ndarray,
                                  shift_inx: list,
                                  keep_non_inx: bool) -> Union[np.ndarray, pd.DataFrame]:
    """
    Converts final shifted_X back into original input type and keeps relevant columns.
    
    JZ 2021
    
    Args:
        X: Original input data
        shifted_X: Post-shift numpy data
        shift_inx: Column numbers that were shifted
        keep_non_inx: Whether or not to keep non-shifted columns in returned value
    
    Returns:
        Shifted values, converted back to the same datatype as X
    """
    if type(X) == pd.DataFrame:
        return_setup = shifted_cols_to_pandas(X, shifted_X, shift_inx, keep_non_inx)
    else:
        return_setup = shifted_cols_to_numpy(X, shifted_X, shift_inx, keep_non_inx)
    return return_setup

def shifted_cols_to_pandas(X: pd.DataFrame,
                           shifted_X: np.ndarray,
                           shift_inx: list,
                           keep_non_inx: bool) -> pd.DataFrame:
    """
    Generate a DataFrame with all shift_inx columns overwritten with the shifted data
    and return either the entire DataFrame or only the subset of columns that have been
    shifted.
    
    JZ 2021
    
    Args:
        X: Original input data
        shifted_X: Post-shift numpy data
        shift_inx: Column numbers that were shifted
        keep_non_inx: Whether or not to keep non-shifted columns in returned value
    
    Returns:
        Shifted values, converted back to a Pandas DataFrame
    """
    return_setup = X.copy()
    return_setup.iloc[:, shift_inx] = shifted_X
    if not keep_non_inx:
        return_setup = return_setup.iloc[:, shift_inx]
    return return_setup

def shifted_cols_to_numpy(X: np.ndarray,
                          shifted_X: np.ndarray,
                          shift_inx: list,
                          keep_non_inx: bool) -> np.ndarray:
    """
    Generate a Numpy Array with all shift_inx columns overwritten with the shifted data
    and return either the entire DataFrame or only the subset of columns that have been
    shifted.
    
    JZ 2021
    
    Args:
        X: Original input data
        shifted_X: Post-shift numpy data
        shift_inx: Column numbers that were shifted
        keep_non_inx: Whether or not to keep non-shifted columns in returned value
    
    Returns:
        Shifted values, converted back to a Numpy Array
    """
    if keep_non_inx:
        return_setup = X.copy()
        return_setup[:, shift_inx] = shifted_X
    else:
        return_setup = shifted_X.copy()
    return return_setup

def concat_all_shifts(X: Union[np.ndarray, pd.DataFrame],
                      shift_amt_list: List[int],
                      shifted_list: List[Union[np.ndarray, pd.DataFrame]]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Concatenate and returns all shifted values in a datatype-relevant way
    (i.e. for DataFrames, rename columns by shift amount, for Numpy Arrays,
    simply concatenate).
    
    JZ 2021
    
    Args:
        X: Original input data
        shift_amt_list: List of timeshifts used
        shifted_list: List of all post-shift values
    
    Returns:
        Final concatenated (column-wise) dataset of all timeshifts
    """
    if type(X) == pd.DataFrame:
        return concat_pandas_shifts(shift_amt_list, shifted_list)
    else:
        return np.concatenate(shifted_list, axis=1)

def concat_pandas_shifts(shift_amt_list: List[int],
                         shifted_list: List[Union[np.ndarray, pd.DataFrame]]
                         ) -> Union[np.ndarray, pd.DataFrame]:
    """
    Concatenate and returns all shifted Pandas values, with columns renamed by shift amount.
    
    JZ 2021
    
    Args:
        shift_amt_list: List of timeshifts used
        shifted_list: List of all post-shift values
    
    Returns:
        Final concatenated (column-wise) dataset of all timeshifts
    """
    # cols = []
    # for _ in shifted_list:
    #     cols.extend(list(_.columns))
    
    # ret = pd.DataFrame()
    ret = []
    for isa, shift_amt in enumerate(shift_amt_list):
        col_names = shifted_list[isa].columns
        sft_col_names = [f"{_}_{shift_amt}" for _ in col_names] if shift_amt != 0 else col_names
        # ret[sft_col_names] = shifted_list[isa][col_names].copy()
        ret.append(shifted_list[isa][col_names].rename({col_names[i]:sft_col_names[i] for i in range(len(sft_col_names))}, axis=1))
    ret = pd.concat(ret, axis=1)
    return ret

def min_max_scale(X: float, lower_bound: float, upper_bound: float) -> Union[np.ndarray, pd.DataFrame]:
    """
    Scale values of X based on lower and upper bounds.
    
    JZ 2020
    
    Args:
        X: Input data
        lower_bound: Lower bound of scaling
        upper_bound: Upper bound of scaling
    
    Returns:
        Scaled value
    """
    return (X - lower_bound) / (upper_bound - lower_bound)

def lambda_min_max(X: pd.Series) -> float:
    """
    Lambda function for min_max_scale to be used with apply. Returns scaled value of center data point by 5th & 95th percentiles.
    
    JZ 2020

    Args:
        X: Input data
    
    Returns:
        Scaled value
    """
    lower_bound = X.quantile(0.05) # X.min()
    upper_bound = X.quantile(0.95) # X.max()
    return min_max_scale(X.iloc[(len(X) + 1)//2 - 1], lower_bound, upper_bound)

    # return min_max_scale(X.iloc[-1], X.min(), X.max())

def detrend_data(X: pd.DataFrame, detrend_col: str, grouping_cols: List[str], window: int, standardize: Optional[bool]=False) -> pd.DataFrame:
    """
    Detrend data by removing a trend from a column.
    
    JZ 2020
    
    Args:
        X: Input data
        detrend_col: Column to detrend
        grouping_cols: Columns to group by
        window: Window size for detrending
    
    Returns:
        Detrended data
    """
    if grouping_cols:
        ret = X.groupby(grouping_cols)[detrend_col].rolling(window=window*2, center=True).apply(lambda_min_max)
    else:
        ret = X[detrend_col].rolling(window=window*2, center=True).apply(lambda_min_max)

    # if standardize:
    #     ret = (ret - ret.mean()) / ret.std()
    
    return ret