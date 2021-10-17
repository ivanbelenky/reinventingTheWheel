import numpy as np


def generate_idxs(shape):
    """Returns list with all indexes created from shape tuple.
    This functions just returns the result of the cartesian product
    of the intervals defined from 0 to n_d for each d dimension represented
    in the shape."""

    intervals = [np.array([i for i in range(size)]) for size in shape]
    out = cartesian(intervals, out=None)
    idxs = [tuple(idx) for idx in out]
    return idxs

import numpy as np

def cartesian(arrays, out=None):
    """
    Implemented by CT Zhu. Reference down below.
    https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    
    
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def filter_idx(idx1, idx2, n):
    new_idx = []
    target_idx = idx2[:n-1]
    for idx in list(idx1):
        if idx[:n-1] == target_idx:
            new_idx.append(tuple(idx))
    return new_idx

