import numpy as np
import random
import matplotlib.pyplot as plt

def ransac_polyfit(x_array_1d, y_array_1d, poly_deg,
                  min_sample_num,
                  residual_thres,
                  max_trials,
                  stop_sample_num=np.inf,
                  random_seed=None):
    """
    Run the RANSAC polynomial fitting.

    Parameters
    ----------
    x_array_1d : ndarray
        xdata.
    x_array_1d : ndarray
        ydata.
    poly_deg : int
        Polynomial degree.
    min_sample_num : int
        Minimum samples numbers needed for RANSAC fitting.
    residual_thres : float
        Residual threshold for RANSAC fitting.
    max_trials : int
        Maximum trials number for RANSAC fitting.
    stop_sample_num : int, optional
        Stop sample number for RANSAC fitting.
    random_seed : float, optional
        Random seed for RANSAC fitting.

    Returns
    -------
    params_tuple_1d: tuple
        1d tuple of the output parameters.

    Examples
    --------
    import numpy as np
    from cellquantifier.qmath.ransac import ransac_polyfit
    a = np.array(range(10))
    b = 1 + 2*a + 3*a ** 2
    params = ransac_polyfit(x_array_1d=a, y_array_1d=b, poly_deg=2,
                    min_sample_num=5, residual_thres=2, max_trials=100)
    print(params)
    """

    # """
    # ~~~~~~~Generate a list. xdata as the 1st col, ydata as the 2nd col~~~~~~~
    # """

    data = np.zeros((len(x_array_1d), 2))
    data[:,0] = x_array_1d
    data[:,1] = y_array_1d
    datalist = list(data)

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~Do the ransac fitting~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """

    best_inlier_count = 0
    best_model = None
    best_err = None
    random.seed(random_seed)
    for i in range(max_trials):
        sample = random.sample(datalist, int(min_sample_num))
        sample_ndarray = np.array(sample)
        xdata_s = sample_ndarray[:,0]
        ydata_s = sample_ndarray[:,1]
        poly_params = np.polyfit(xdata_s, ydata_s, poly_deg)
        p = np.poly1d(poly_params)

        inlier_count = 0
        for j in range(len(data)):
            curr_x = data[j,0]
            rms = np.abs(p(curr_x) - data[j, 1])
            if rms < residual_thres:
                inlier_count = inlier_count + 1

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_model = poly_params
            if inlier_count > stop_sample_num:
                break

    # """
    # ~~~~~~~~~~~~~~~~~~~~~Print the ransac fitting summary~~~~~~~~~~~~~~~~~~~~~
    # """

    print("#" * 30)
    print("Iteration_num: ", i+1)
    print("Best_inlier_count: ", best_inlier_count)
    print("Best_model: ", best_model)
    print("#" * 30)

    params_tuple_1d = best_model

    return params_tuple_1d
