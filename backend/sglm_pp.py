import numpy as np
import pandas as pd
import scipy.signal
from numba import njit, jit, prange

import sys
sys.path.append('./lib/CaImAn/caiman/source_extraction/cnmf')

from deconvolution import constrained_foopsi

# TODO: Write testcases & check validity

# TODO: Include train/test split - by 2 min segmentation
# TODO: Include diff

# TODO: specify np.ascontiguousarray

# Replace deconvolve with https://github.com/agiovann/constrained_foopsi_python


def timeshift(X, shift_inx=[], shift_amt=1, keep_non_inx=False):
    """
    Shift the column indicies "shift_inx" forward by shift_amt steps (backward if shift_amt < 0)

    Parameters
    ----------
    X : np.ndarray (preferably contiguous array)
        Array of all variables (columns should be features, rows should be timesteps)
    shift_inx : list(int)
        Column indices to shift forward/backward
    shift_amt : int
        Amount by which to shift forward the columns in question (backward if shift_amt < 0)
    keep_non_inx : bool
        If True, data from all columns (shifted or not) will be returned from the function. If False,
        only shifted columns are returned.
    """

    if type(X) == pd.DataFrame:
        npX = X.values
    else:
        npX = X
    
    shift_inx = shift_inx if shift_inx else range(npX.shape[1])
    X_to_shift = npX[:, shift_inx]

    append_vals = np.zeros((np.abs(shift_amt), X_to_shift.shape[1]))
    if shift_amt > 0:
        shifted_X = np.concatenate((append_vals, X_to_shift), axis=0)
        shifted_X = shifted_X[:-np.abs(shift_amt), :]
    elif shift_amt < 0:
        shifted_X = np.concatenate((X_to_shift, append_vals), axis=0)
        shifted_X = shifted_X[np.abs(shift_amt):, :]
    else:
        shifted_X = X_to_shift
    
    if type(X) == pd.DataFrame:
        return_setup = X.copy()
        return_setup.iloc[:, shift_inx] = shifted_X
        if not keep_non_inx:
            return_setup = return_setup.iloc[:, shift_inx]
    else:
        if keep_non_inx:
            return_setup = npX.copy()
            return_setup[:, shift_inx] = shifted_X
        else:
            return_setup = shifted_X.copy()

    return return_setup


def timeshift_multiple(X, shift_inx=[], shift_amt_list=[-1,0,1], unshifted_keep_all=True):
    """
    Collect all forward/backward shifts of columns shift_inx as columns in the returned array

    Parameters
    ----------
    X : np.ndarray (preferably contiguous array)
        Array of all variables (columns should be features, rows should be timesteps)
    shift_inx : list(int)
        Column indices to shift forward/backward
    shift_amt_list : list(int)
        List of amounts by which to shift forward the columns in question (backward where list elements are < 0)
    unshifted_keep_all : bool
        Whether or not to keep all unshifted columns in the returned array
    """

    shifted_list = []
    for shift_amt in shift_amt_list:
        shifted = timeshift(X, shift_inx=shift_inx, shift_amt=shift_amt, keep_non_inx=(shift_amt == 0 and unshifted_keep_all))
        shifted_list.append(shifted)
    
    return np.concatenate(shifted_list, axis=1)


def zscore(X):
    """
    Convert X values to z-scores along the 0th axis

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Array of variables to zscore
    """
    return (X - X.mean(axis=0))/X.std(axis=0)


def deconvolve(*args, **kwargs):
    """
    Deconvolve using CaImAn implementation of constrained_foopsi. (Documentation follows.)
    ---

    Infer the most likely discretized spike train underlying a fluorescence trace

    It relies on a noise constrained deconvolution approach
    
    Parameters
    ----------
        fluor: np.ndarray
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.

        bl: [optional] float
            Fluorescence baseline value. If no value is given, then bl is estimated
            from the data.

        c1: [optional] float
            value of calcium at time 0

        g: [optional] list,float
            Parameters of the AR process that models the fluorescence impulse response.
            Estimated from the data if no value is given

        sn: float, optional
            Standard deviation of the noise distribution.  If no value is given,
            then sn is estimated from the data.

        p: int
            order of the autoregression model

        method_deconvolution: [optional] string
            solution method for basis projection pursuit 'cvx' or 'cvxpy' or 'oasis'

        bas_nonneg: bool
            baseline strictly non-negative

        noise_range:  list of two elms
            frequency range for averaging noise PSD

        noise_method: string
            method of averaging noise PSD

        lags: int
            number of lags for estimating time constants

        fudge_factor: float
            fudge factor for reducing time constant bias

        verbosity: bool
             display optimization details

        solvers: list string
            primary and secondary (if problem unfeasible for approx solution) solvers
            to be used with cvxpy, default is ['ECOS','SCS']

        optimize_g : [optional] int, only applies to method 'oasis'
            Number of large, isolated events to consider for optimizing g.
            If optimize_g=0 (default) the provided or estimated g is not further optimized.

        s_min : float, optional, only applies to method 'oasis'
            Minimal non-zero activity within each bin (minimal 'spike size').
            For negative values the threshold is abs(s_min) * sn * sqrt(1-g)
            If None (default) the standard L1 penalty is used
            If 0 the threshold is determined automatically such that RSS <= sn^2 T

    Returns:
        c: np.ndarray float
            The inferred denoised fluorescence signal at each time-bin.

        bl, c1, g, sn : As explained above

        sp: ndarray of float
            Discretized deconvolved neural activity (spikes)

        lam: float
            Regularization parameter
    Raises:
        Exception("You must specify the value of p")

        Exception('OASIS is currently only implemented for p=1 and p=2')

        Exception('Undefined Deconvolution Method')

    References:
        * Pnevmatikakis et al. 2016. Neuron, in press, http://dx.doi.org/10.1016/j.neuron.2015.11.037
        * Machado et al. 2015. Cell 162(2):338-350

    \image: docs/img/deconvolution.png
    \image: docs/img/evaluationcomponent.png
    """

    return constrained_foopsi(*args, **kwargs)

def cvt_to_contiguous(x):
    return x

@jit(parallel=True)
def zscore_numba(array):
    '''
    Parallel (multicore) Z-Score. Uses numba.
    Computes along second dimension (axis=1) for speed
    Best to input a contiguous array.
    RH 2021
    Args:
        array (ndarray):
            2-D array. Percentile will be calculated
            along first dimension (columns)
    
    Returns:
        output_array (ndarray):
            2-D array. Z-Scored array
    '''

    output_array = np.zeros_like(array)
    for ii in prange(array.shape[0]):
        array_tmp = array[ii,:]
        output_array[ii,:] = (array_tmp - np.mean(array_tmp)) / np.std(array_tmp)
    return output_array


# """
# Credit: Rich Hakim
# """
# @njit(parallel=True)
# def var_numba(X):
#     Y = np.zeros(X.shape[0], dtype=X.dtype)
#     for ii in prange(X.shape[0]):
#         Y[ii] = np.var(X[ii,:])
#     return Y


# @njit(parallel=True)
# def min_numba(X):
#     output = np.zeros(X.shape[0])
#     for ii in prange(X.shape[0]):
#         output[ii] = np.min(X[ii])
#     return output


# @njit(parallel=True)
# def max_numba(X):
#     output = np.zeros(X.shape[0])
#     for ii in prange(X.shape[0]):
#         output[ii] = np.max(X[ii])
#     return output

