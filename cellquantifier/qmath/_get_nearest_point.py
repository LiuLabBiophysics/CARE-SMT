import numpy as np

def get_nearest_point(X, X_array):
    """
    Find the nearest point with point X among points in X_array

    Pseudo code
    ----------
    1. Calculate distance array between X_array and X.
    2. Find the index of minimum distance.
    3. Find the nearest point in X_array using the index.

    Parameters
    ----------
    X : numpy ndarray
        This list defines the point position
    X_array : numpy darray
        List of points to be tested

    Returns
    -------
    nearest_point : numpy array
        The nearest point.

    Examples
	--------
    from cellquantifier.qmath import *
    import numpy as np
    X = np.array([1.5, 1.5])
    X_array = np.array([[0,0], [1,1], [2,2]])
    print(get_nearest_point(X, X_array))
    """
    dist_array = (X_array - X)**2
    dist_array = dist_array.sum(axis=1)
    ind = np.argmin(dist_array)
    nearest_point = X_array[ind]

    return nearest_point
