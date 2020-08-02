import cython

cimport numpy as np
import numpy as np 
import math

# -*- coding: utf-8 -*-
"""Lempel-Ziv complexity for a binary sequence, in simple Cython code (C extension).
- How to build it? Simply use the file :download:`Makefile` provided in this folder.
- How to use it? From Python, it's easy:
>>> from lempel_ziv_complexity_cython import lempel_ziv_complexity
>>> s = '1001111011000010'
>>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010
8
- Requirements: you need to have [Cython](http://Cython.org/) installed, and use [CPython](https://www.Python.org/).
- MIT Licensed, (C) 2017-2019 Lilian Besson (Naereen)
  https://GitHub.com/Naereen/Lempel-Ziv_Complexity
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

# Define the type of unsigned int32
ctypedef unsigned int DTYPE_t

# turn off bounds-checking for entire function, quicker but less safe
@cython.boundscheck(False)
def lempel_ziv_complexity(str sequence):
    """Lempel-Ziv complexity for a binary sequence, in simple Cython code (C extension).
    It is defined as the number of different substrings encountered as the stream is viewed from begining to the end.
    As an example:
    >>> s = '1001111011000010'
    >>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010
    8
    Marking in the different substrings the sequence complexity :math:`\mathrm{Lempel-Ziv}(s) = 8`: :math:`s = 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010`.
    - See the page https://en.wikipedia.org/wiki/Lempel-Ziv_complexity for more details.
    Other examples:
    >>> lempel_ziv_complexity('1010101010101010')  # 1, 0, 10, 101, 01, 010, 1010
    7
    >>> lempel_ziv_complexity('1001111011000010000010')  # 1, 0, 01, 11, 10, 110, 00, 010, 000
    9
    >>> lempel_ziv_complexity('100111101100001000001010')  # 1, 0, 01, 11, 10, 110, 00, 010, 000, 0101
    10
    - Note: it is faster to give the sequence as a string of characters, like `'10001001'`, instead of a list or a numpy array.
    - Note: see this notebook for more details, comparison, benchmarks and experiments: https://Nbviewer.Jupyter.org/github/Naereen/Lempel-Ziv_Complexity/Short_study_of_the_Lempel-Ziv_complexity.ipynb
    - Note: there is also a naive Python version, for speedup, see :download:`lempel_ziv_complexity.py`.
    """
    cdef set sub_strings = set()
    cdef str sub_str = ""
    cdef DTYPE_t n = len(sequence)
    cdef DTYPE_t ind = 0
    cdef DTYPE_t inc = 1
    while True:
        if ind + inc > len(sequence):
            break
        sub_str = sequence[ind : ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    #normalized LZ complexity of binary sequences as found in Lempel & Ziv:
    #On the Complexity of Finite Sequences,
    #IEEE Transactions on Information Theory Vol. IT-22 No. 1, Jan 1976
    return len(sub_strings) * math.log(n, 2) / n

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cython_approximate_entropy(double[:] timeseries, int m, double r):
    """
    Approximate entropy for a given time series, calculated with window size m and noise level r.
    Speed-up of 800x over the naive Python version 
    """

    cdef Py_ssize_t N = timeseries.shape[0]
    cdef Py_ssize_t num_runs = 2
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t window, vec_len

    cdef double[:,::1] x
    cdef double[::1] C
    cdef double cand
    cdef double[2] phi = [0.0, 0.0] 

    for k in range(num_runs):
        window = m + k

        vec_len = N - window + 1
        norm = 1.0 / (<double>vec_len * math.log(2))

        x = np.zeros((vec_len, window), dtype = np.double)

        for i in range(vec_len):
            for j in range(window):
                x[i,j] = timeseries[i + j]

        #ones because of self-counts
        C = np.ones(vec_len, dtype = np.double)

        for i in range(vec_len):
            for j in range(i+1, vec_len): 
                cand = <double>(cython_max_norm(x, i, j) < r)
                C[i] += cand
                C[j] += cand
            C[i] *= norm

        phi[k] = np.sum(np.log(C)) * norm

    return abs(phi[0] - phi[1])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cython_sample_entropy(double[:] timeseries, int m, double r):
    """
    Approximate entropy for a given time series, calculated with window size m and noise level r.
    Speed-up of 800x over the naive Python version 
    """

    cdef Py_ssize_t N = timeseries.shape[0]
    cdef Py_ssize_t num_runs = 2
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t window, vec_len

    cdef double[:,::1] x
    cdef int C

    cdef double[2] phi = [0.0, 0.0] 

    for k in range(num_runs):
        window = m + k

        vec_len = N - window + 1

        x = np.zeros((vec_len, window), dtype = np.double)

        for i in range(vec_len):
            for j in range(window):
                x[i,j] = timeseries[i + j]

        #we don't need to save the intermediate values 
        C = 0

        for i in range(vec_len-1):
            for j in range(i+1, vec_len): 
                C += 2 * <int>(cython_max_norm(x, i, j) < r)

        if C == 0:
            return np.nan

        phi[k] = <double>C
        
    return np.log2(phi[0] / phi[1])

@cython.boundscheck(False)
@cython.wraparound(False)
#putting cpdef instead of def cuts the time by a factor of about 8!!!
cpdef double cython_max_norm(double[:, ::1] X, Py_ssize_t i, Py_ssize_t j):
    cdef Py_ssize_t k
    cdef double max_norm = 0.0
    cdef Py_ssize_t window = X.shape[1]
    cdef double diff

    for k in range(window):
        diff = abs(X[i, k] - X[j, k])
        if diff > max_norm: 
            max_norm = diff
    
    return max_norm