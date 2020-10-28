from ._add_hist import *

def add_D_hist(ax, df,
            cat_col=None,
            cat_order=None,
            color_list=None,
            RGBA_alpha=0.5,
            hist_kws=None,
            kde=True,
            set_format=True):
    """
    Add D histogram in matplotlib axis.

    Pseudo code
    ----------
    1. If df is empty, return.
    2. Drop duplicates D values based on 'particle' column.
    3. Add D histograms to the ax.
    4. Format the ax if needed.

    Parameters
    ----------
    ax : object
        matplotlib axis.

    df : DataFrame
		DataFrame contains with columns 'D', 'particle'
        may contain cat_col.

    data_col : str
        Column used to plot histogram.

    cat_col : str, optional
		Column to use for categorical sorting.

    cat_order : list of str, optional
        Prefered order to plot the histograms.

    color_list : list of tuple/list, optional
        Prefered colors to plot histograms.

    RGBA_alpha : float, optional
        If alpha value is not specified, this value will be used.

    set_format : bool, optional
        If true, set the ax format to default format.

    Returns
    -------
    Add D histograms in the ax.

    Examples
	--------
    import matplotlib.pyplot as plt
    import pandas as pd
    from cellquantifier.plot.plotutil import *
    filepath = 'cellquantifier/data/physDataMerged.csv'
    df = pd.read_csv(filepath, index_col=None, header=0)
    df = df.drop_duplicates('particle')
    fig, ax = plt.subplots()
    add_D_hist(ax, df,
            cat_col='exp_label',
            cat_order=['Ctr', 'BLM'],
            RGBA_alpha=0.5)
    plt.show()
    """

    # """
    # ~~~~~~~~~~~Check if df is empty~~~~~~~~~~~~
    # """
    if df.empty:
    	return

    # """
    # ~~~~Drop duplicates D values based on 'particle' column~~~~
    # """
    df = df.drop_duplicates('particle')

    # """
    # ~~~~Add D histograms to the ax~~~~
    # """
    add_hist(ax, df, 'D',
            cat_col=cat_col,
            cat_order=cat_order,
            color_list=color_list,
            RGBA_alpha=RGBA_alpha,
            hist_kws=hist_kws,
            kde=kde)

    # """
    # ~~~~Format the ax if needed~~~~
    # """
    if set_format:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(labelsize=13, width=2, length=5)
        ax.get_yaxis().set_ticks([])

        ax.set_ylabel(r'$\mathbf{PDF}$', fontsize=15)
        ax.set_xlabel(r'$\mathbf{D (nm^{2}/s)}$', fontsize=15)
        ax.legend()
