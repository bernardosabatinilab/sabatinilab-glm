from typing import DefaultDict, List, TypeVar, Optional, Union, List, Tuple
import numpy as np
import pandas as pd

def shift_series_range(series: pd.Series, shift_amt_range: Tuple[int], fill_value: Optional[float] = np.nan, shift_bounding_column: Optional[str] = None) -> pd.Series:
    """
    Shift all of series up/down by every value between shift_amt_range[0] and shift_amt_range[1]
    
    JZ 2023

    Args:
        series (pd.Series): Series to be shifted up or down
        shift_amt_range (Tuple[int]): Range of amounts to shift data up or down (> 0 = shift down, < 0 = shift up)
        fill_value (Optional[float]): Value to be left in place of shifted data (default: np.nan)

    Returns:
        df_shifted_series (pd.DataFrame): Concatenated post-shift versions of series
    """
    list_shifted_series = []
    for ishift_amt in range(shift_amt_range[0], shift_amt_range[1]+1):
        series_shifted = shift_series(series, ishift_amt, fill_value=fill_value, shift_bounding_column=shift_bounding_column).rename((f"{series.name}", f"{ishift_amt}"))
        list_shifted_series.append(series_shifted)
    df_shifted_series = pd.concat(list_shifted_series, axis=1)
    return df_shifted_series

def shift_series(series: pd.Series, shift_amt: int, fill_value: Optional[float] = np.nan, shift_bounding_column: Optional[str] = None) -> pd.Series:
    """
    Shift all of series up or down by shift_amt (if > 0: shift down, if < 0: shift up)
    
    JZ 2023
    
    Args:
        series: Series to be shifted up or down
        shift_amt: Amount to shift data up or down (> 0 = shift down, < 0 = shift up)
        fill_value: Optional; Value to be left in place of shifted data
    
    Returns:
        shifted_series : Post-shift version of series
    """
    series_postGroup = series.groupby(shift_bounding_column) if shift_bounding_column is not None else series
    if shift_amt > 0:
        shifted_series = series_postGroup.shift(periods=shift_amt, fill_value=fill_value)
    elif shift_amt < 0:
        shifted_series = series_postGroup.shift(periods=shift_amt, fill_value=fill_value)
    else:
        shifted_series = series
    return shifted_series

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