"""Matrix Profile Distances."""

from numba import njit
import numpy as np


@njit(cache=True, fastmath=True)
def mpdist(
    x: np.ndarray,
    y: np.ndarray,
    m: int = 0
) -> float:
    r"""Compute the Matrix Profile distance between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    m : int
        Length of the subsequence.

    Returns
    -------
    float
        Matrix Profile distance between x and y, minimum value 0.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] Ratanamahatana C and Keogh E.: Three myths about dynamic time warping data
    mining, Proceedings of 5th SIAM International Conference on Data Mining, 2005.

    .. [2] Sakoe H. and Chiba S.: Dynamic programming algorithm optimization for
    spoken word recognition. IEEE Transactions on Acoustics, Speech, and Signal
    Processing 26(1):43–49, 1978.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import mpdist
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    >>> mpdist(x, y) # 1D series
    768.0
    >>> x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 1, 0, 2, 0]])
    >>> y = np.array([[11, 12, 13, 14],[7, 8, 9, 20],[1, 3, 4, 5]] )
    >>> mpdist(x, y) # 2D series with 3 channels, unequal length
    564.0
    """
    if x.ndim == 2 and x.shape[0] > 1:
        raise ValueError("x must be a 1D array or shape (1,n)")
    if y.ndim == 2 and y.shape[0] > 1:
        raise ValueError("y must be a 1D array or shape (1,n)")
    x = x.squeeze()
    y = y.squeeze()

    len_x = len(x)
    len_y = len(y)

    if m == 0:
        if len1 > len2:
            m = int(len1 / 4)
        else:
            m = int(len2 / 4)
    threshold = 0.05
    mp_ab, ip_ab = _stomp_ab(x, y, m)  # compute the AB matrix profile
    mp_ba, ip_ba = _stomp_ab(y, x, m)  # compute the BA matrix profile

    join_mp = np.concatenate([mp_ab, mp_ba])

    k = int(np.ceil(threshold * (len_x + len_y)))

    sorted_mp = np.sort(join_mp)  # sort the join matrix profile in ascending order

    if len(sorted_mp) > k:
        mpdist = sorted_mp[k]
    else:
        mpdist = sorted_mp[len(sorted_mp) - 1]

    return mpdist


@njit(cache=True, fastmath=True)
def _sliding_dot_products(q, t, q_len, t_len):
    """
    Compute the sliding dot products between a query and a time series.

    Parameters
    ----------
        q: numpy.array
            Query.
        t: numpy.array
            Time series.
        q_len: int
            Length of the query.
        t_len: int
            Length of the time series.

    Output
    ------
        dot_prod: numpy.array
                    Sliding dot products between q and t.
    """
    # Reversing query and padding both query and time series
    t_padded = np.pad(t, (0, t_len))
    q_reversed = np.flipud(q)
    q_reversed_padded = np.pad(q_reversed, (0, 2 * t_len - q_len))

    # Applying FFT to both query and time series
    t_fft = np.fft.fft(t_padded)
    q_fft = np.fft.fft(q_reversed_padded)

    # Applying inverse FFT to obtain the convolution of the time series by
    # the query
    element_wise_mult = np.multiply(t_fft, q_fft)
    inverse_fft = np.fft.ifft(element_wise_mult)

    # Returns only the valid dot products from inverse_fft
    dot_prod = inverse_fft[q_len - 1 : t_len].real

    return dot_prod


@njit(cache=True, fastmath=True)
def _calculate_distance_profile(
    dot_prod, q_mean, q_std, t_mean, t_std, q_len, n_t_subs
):
    """
    Calculate the distance profile for the given query.

    Parameters
    ----------
        dot_prod: numpy.array
            Sliding dot products between the time series and the query.
        q_mean: float
            Mean of the elements of the query.
        q_std: float
            Standard deviation of elements of the query.
        t_mean: numpy.array
            Array with the mean of the elements from each subsequence of
            length(query) from the time series.
        t_std: numpy.array
            Array with the standard deviation of the elements from each
            subsequence of length(query) from the time series.
        q_len: int
            Length of the query.
        n_t_subs: int
            Number of subsequences in the time series.

    Output
    ------
        d: numpy.array
            Distance profile of query q.
    """
    d = [
        2
        * q_len
        * (
            1
            - ((dot_prod[i] - q_len * q_mean * t_mean[i]) / (q_len * q_std * t_std[i]))
        )
        for i in range(0, n_t_subs)
    ]
    d = np.absolute(d)
    d = np.sqrt(d)

    return d


@njit(cache=True, fastmath=True)
def _stomp_ab(x, y, m):
    """
    STOMP implementation for AB similarity join.

    Parameters
    ----------
        x: numpy.array
            First time series.
        y: numpy.array
            Second time series.
        m: int
            Length of the subsequences.

    Output
    ------
        mp: numpy.array
            Array with the distance between every subsequence from x
            to the nearest subsequence with same length from y.
        ip: numpy.array
            Array with the index of the nearest neighbor of x in y.
    """
    len_x = len(x)
    len_y = len(y)

    # Number of subsequences
    n_x_subs = len_x - m + 1
    n_y_subs = len_y - m + 1

    # Compute the mean and standard deviation
    x_mean = [np.mean(x[i : i + m]) for i in range(0, n_x_subs)]
    x_std = [np.std(x[i : i + m]) for i in range(0, n_x_subs)]

    y_mean = [np.mean(y[i : i + m]) for i in range(0, n_y_subs)]
    y_std = [np.std(y[i : i + m]) for i in range(0, n_y_subs)]

    # Compute the dot products between the first ts2 subsequence and every
    # ts1 subsequence
    dot_prod = _sliding_dot_products(y[0:m], x, m, len_x)
    first_dot_prod = np.copy(dot_prod)

    # Initialization
    mp = np.full(n_x_subs, float("inf"))  # matrix profile
    ip = np.zeros(n_x_subs)  # index profile

    # Compute the distance profile for the first ts1 subsequence
    dot_prod = _sliding_dot_products(x[0:m], y, m, len_y)
    dp = _calculate_distance_profile(
        dot_prod, x_mean[0], x_std[0], y_mean, y_std, m, n_y_subs
    )

    # Updates the matrix profile
    mp[0] = np.amin(dp)
    ip[0] = np.argmin(dp)

    for i in range(1, n_x_subs):
        for j in range(n_y_subs - 1, 0, -1):
            dot_prod[j] = (
                dot_prod[j - 1]
                - y[j - 1] * x[i - 1]
                + y[j - 1 + m] * x[i - 1 + m]
            )  # compute the next dot products
            # using the previous ones
        dot_prod[0] = first_dot_prod[i]
        dp = _calculate_distance_profile(
            dot_prod, x_mean[i], x_std[i], y_mean, y_std, m, n_y_subs
        )
        mp[i] = np.amin(dp)
        ip[i] = np.argmin(dp)

    return mp, ip
