import numpy as np
from joblib import Parallel, delayed


def resample_estimate(*data, estimator, resample, n_resamples=1000, n_jobs=5):
    """ Resample and compute the estimator.
    This implementation enable parallelization using joblib and return an array
    of n_perms x nb_statistics. 

    Parameters
    ----------
    data : iterable of array-like, at least one array.
        Contains the samples, each of which is an 2D array of N observations.
        Axis 0 of sample arrays must the same.
    estimator : callable function
        Statistic for which the p-value of the hypothesis test is to be
        calculated. Accepts iterable of array-like and returns `nb_statistics`.
    resample : Callable function
        Data resampling scheme. Accepts iterable of array-like and returns
        resampled iterable of array-like.
    n_resamples : int
        number of resampling (default 1000)
    n_jobs : int
        number of parallel jobs  (default 5)
        
    Returns
    -------
    Array of shape (n_perms, nb_statistics). nb_statistics being the 
    nb_statistics (size) returned by callable `estimator`.
    
    Examples
    --------
    >>> from joblib import Parallel, delayed
    >>> import numpy as np
    >>> def id_permutation(*data):
    >>>    return data
    >>> def mean(*data):
    >>>    return np.mean(data[0], axis=0)
    >>> x = np.arange(5)
    >>> stats_rnd = resample_estimate(x, estimator=mean,
        resample=id_permutation, n_perms=2)
    >>> print(stats_rnd)
    [[2.]
     [2.]]
    """

    # Callable executed for each permutation
    def resample_estimate_(*data, estimator, resample):
        return  estimator(*resample(*data))  # Statistics on Resampled data
     
    parallel = Parallel(n_jobs=n_jobs)
    stats_rnd = parallel(
        delayed(resample_estimate_)(
            *data, estimator=estimator, resample=resample)
        for r in range(n_resamples))
    return np.array(stats_rnd)
