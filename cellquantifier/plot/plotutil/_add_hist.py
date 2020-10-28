import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cellquantifier.io import *

def add_hist(ax, df, data_col,
            cat_col=None,
            cat_order=None,
            color_list=None,
            RGBA_alpha=1,
            bins=None,
            hist_kws=None,
            kde=True):
    """
    Add histogram in matplotlib axis.

    Pseudo code
    ----------
    1. If df is empty, return.
    2. Prepare the cats based on (cat_col, cat_order).
    3. Prepare the colors based on (cat_col, color_list).
    4. Add histgrams accordingly.

    Parameters
    ----------
    ax : object
        matplotlib axis.

    df : DataFrame
		DataFrame containing necessary data.

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

    Returns
    -------
    Add histograms in the ax.

    Examples
	--------
    import matplotlib.pyplot as plt
    import pandas as pd
    from cellquantifier.plot.plotutil import add_hist
    filepath = 'cellquantifier/data/physDataMerged.csv'
    df = pd.read_csv(filepath, index_col=None, header=0)
    df = df.drop_duplicates('D')
    fig, ax = plt.subplots()
    add_hist(ax, df, 'D',
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
    # ~~~~Prepare the cats based on (cat_col, cat_order)~~~~
    # """
    if (isinstance(cat_order, list) and (check_elem_type(cat_order, str))):
        cats = cat_order
    else:
        if cat_col:
            cats = sorted(df[cat_col].unique())
        else:
            cats = [data_col]

    # """
    # ~~~~Prepare the colors based on (cat_col, color_list)~~~~
    # """
    if (isinstance(color_list, list) \
        and (check_elem_type(color_list, list) or \
            check_elem_type(color_list, tuple)) \
        and (check_elem_length(color_list, 3) or \
            check_elem_length(color_list, 4)) \
        and len(color_list)==len(cats)):
        colors = color_list

    else:
        if cat_col:
            colors = plt.cm.coolwarm(np.linspace(0,1,len(cats)))
            colors[:, 3] = RGBA_alpha
        else:
            colors = [plt.cm.coolwarm(0.99)]

    # """
    # ~~~~~~~~~~~~Add histgrams accordingly~~~~~~~~~~~~
    # """
    for i in range(len(cats)):
        if cat_col != None:
            curr_cat = cats[i]
            curr_df = df.loc[ df[cat_col]==curr_cat ]
            curr_color = tuple(colors[i])
        else:
            curr_cat = cats
            curr_df = df
            curr_color = tuple(colors[i])


        sns.set(style="white", palette="coolwarm", color_codes=True)
        sns.distplot(curr_df[data_col].to_numpy(),
                    bins=bins,
                    hist=True, hist_kws=hist_kws, kde=kde,
                    color=curr_color, label=curr_cat, ax=ax)

    ax.legend()
