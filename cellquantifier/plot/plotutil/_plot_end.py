import matplotlib.pyplot as plt
from ._plt2array import plt2array

def plot_end(fig, pltshow=False):
    """
    Typical piece of code at the end of diagnostic.

    Parameters
    ----------
    fig : object
        matplotlib figure object.
    pltshow: bool, optional
        If True, show the fig.
        If False, clear and close the fig.

    Returns
    -------
    rgb_array_3d: ndarray
        3d ndarray represting the figure
    """

    rgb_array_3d = plt2array(fig)

    plt.tight_layout()
    if pltshow:
        plt.show()
    else:
        plt.clf()
        plt.close()
    return rgb_array_3d
