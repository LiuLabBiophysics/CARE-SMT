import numpy as np
from scipy.optimize import curve_fit

def gaussian_2d(X, A, x0, y0, sig_x, sig_y, phi):
    """
    2D Gaussian function

    Parameters
    ----------
    X : 3d ndarray
        X = np.indices(img.shape).
		X[0] is the row indices.
        Y[1] is the column indices.
    A : float
        Amplitude.
    x0 : float
        x coordinate of the center.
    y0 : float
        y coordinate of the center.
    sig_x : float
        Sigma in x direction.
    sig_y : float
        Sigma in x direction.
    phi : float
        Angle between long axis and x direction.


    Returns
    -------
    result_array_2d: 21d ndarray
        2D gaussian.
    """

    x = X[0]
    y = X[1]
    a = (np.cos(phi)**2)/(2*sig_x**2) + (np.sin(phi)**2)/(2*sig_y**2)
    b = -(np.sin(2*phi))/(4*sig_x**2) + (np.sin(2*phi))/(4*sig_y**2)
    c = (np.sin(phi)**2)/(2*sig_x**2) + (np.cos(phi)**2)/(2*sig_y**2)
    result_array_2d = A*np.exp(-(a*(x-x0)**2+2*b*(x-x0)*(y-y0)+c*(y-y0)**2))

    return result_array_2d

def get_moments(img):
    """
    Get gaussian parameters of a x2D distribution by calculating its moments

    Parameters
    ----------
    img : 2d ndarray
        image.

    Returns
    -------
    params_tuple_1d: tuple
        parameters (A, x0, y0, sig_x, sig_y, phi).
    """

    total = img.sum()
    X, Y = np.indices(img.shape)
    x0 = (X*img).sum()/total
    y0 = (Y*img).sum()/total
    col = img[:, int(y0)]
    sig_x = np.sqrt(np.abs((np.arange(col.size)-y0)**2*col).sum()/col.sum())
    row = img[int(x0), :]
    sig_y = np.sqrt(np.abs((np.arange(row.size)-x0)**2*row).sum()/row.sum())
    A = img.max()
    phi = 0
    params_tuple_1d = A, x0, y0, sig_x, sig_y, phi
    return params_tuple_1d

def fit_gaussian_2d(img, diagnostic=False):
    """
    Fit gaussian_2d

    Parameters
    ----------
    img : 2d ndarray
        image.
    diagnostic : bool, optional
        If True, show the diagnostic plot

    Returns
    -------
    popt, pcov: 1d ndarray
        optimal parameters and covariance matrix

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    from cellquantifier.qmath.gaussian_2d import gaussian_2d, fit_gaussian_2d
    from cellquantifier.io.imshow import imshow
    X = np.indices((100,100))
    A, x0, y0, sig_x, sig_y, phi = 1, 50, 80, 30, 10, 0.174
    out_array_1d = gaussian_2d(X, A, x0, y0, sig_x, sig_y, phi)
    img = out_array_1d.reshape((100,100))
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()
    popt, p_err = fit_gaussian_2d(img, diagnostic=True)
    print(popt)
    """

    # """
    # ~~~~~~~~~~~~~~Prepare the input data and initial conditions~~~~~~~~~~~~~~
    # """

    X = np.indices(img.shape)
    x = np.ravel(X[0])
    y = np.ravel(X[1])
    xdata = np.array([x,y])
    ydata = np.ravel(img)
    p0 = get_moments(img)

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Fitting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """

    popt, pcov = curve_fit(gaussian_2d, xdata, ydata, p0=p0)
    p_sigma = np.sqrt(np.diag(pcov))
    p_err = p_sigma

    # """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Diagnostic~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # """

    if diagnostic:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')

        (A, x0, y0, sig_x, sig_y, phi) = popt
        (A_err, x0_err, y0_err, sigma_x_err, sigma_y_err, phi_err) = p_err
        Fitting_data = gaussian_2d(X,A,x0,y0,sig_x,sig_y,phi)
        ax.contour(Fitting_data, cmap='cool')
        ax.text(0.95,
                0.00,
                """
                x0: %.3f (\u00B1%.3f)
                y0: %.3f (\u00B1%.3f)
                sig_x: %.3f (\u00B1%.3f)
                sig_y: %.3f (\u00B1%.3f)
                phi: %.1f (\u00B1%.2f)
                """ %(x0, x0_err,
                      y0, y0_err,
                      sig_x, sigma_x_err,
                      sig_y, sigma_y_err,
                      np.rad2deg(phi), np.rad2deg(phi_err)),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize = 12,
                color = (1, 1, 1, 0.8),
                transform=ax.transAxes)
        plt.show()

    return popt, p_err
