import matplotlib.pyplot as plt
import numpy as np

def plt2array(fig):
    """
    Save matplotlib.pyplot figure to numpy rgbndarray.

    Parameters
    ----------
    fig : object
        matplotlib figure object.

    Returns
    -------
    rgb_array_rgb: ndarray
        3d ndarray represting the figure

    Examples
    --------
    import matplotlib.pyplot as plt
    import numpy as np
    from cellquantifier.plot.plotutil import plt2array
    t = np.linspace(0, 4*np.pi, 1000)
    fig, ax = plt.subplots()
    ax.plot(t, np.cos(t))
    ax.plot(t, np.sin(t))
    result_array_rgb = plt2array(fig)
    plt.clf()
    plt.close()
    print(result_array_rgb.shape)
    """

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    rgb_array_rgb = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

    return rgb_array_rgb
