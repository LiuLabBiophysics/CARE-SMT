import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp

from ...qmath import msd, fit_msd

def add_mean_msd(ax, df,
                pixel_size,
                frame_rate,
                divide_num,
                cat_col=None,
                cat_order=None,
                RGBA_alpha=0.5,
                fitting_linewidth=3,
                elinewidth=None,
                markersize=None,
                capsize=2,
                set_format=True):
    """
    Add mean MSD curve in matplotlib axis.
    The MSD data are obtained from df.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.

    df : DataFrame
		DataFrame containing 'particle', 'frame', 'x', and 'y' columns

    cat_col : None or str
		Column to use for categorical sorting


    Returns
    -------
    Annotate mean MSD in the ax.

    Examples
	--------
	import matplotlib.pyplot as plt
    import pandas as pd
    from cellquantifier.plot.plotutil import add_mean_msd
    path = 'cellquantifier/data/physDataMerged.csv'
    df = pd.read_csv(path, index_col=None, header=0)
    fig, ax = plt.subplots()
    add_mean_msd(ax, df, 'exp_label',
                pixel_size=0.108,
                frame_rate=3.33,
                divide_num=5,
                RGBA_alpha=0.5)
    plt.show()
    """


    # """
    # ~~~~~~~~~~~Check if df is empty~~~~~~~~~~~~~~
    # """

    if df.empty:
    	return


    # """
    # ~~~~~~~~~~~Prepare the data, category, color~~~~~~~~~~~~~~
    # """
    if cat_col:
        if cat_order:
            cats = cat_order
        else:
            cats = sorted(df[cat_col].unique())
        dfs = [df.loc[df[cat_col] == cat] for cat in cats]
        colors = plt.cm.coolwarm(np.linspace(0,1,len(cats)))
        colors[:, 3] = RGBA_alpha

        cats_label = cats.copy()
        if 'sort_flag_' in cat_col:
            for m in range(len(cats_label)):
                cats_label[m] = cat_col[len('sort_flag_'):] + ': ' + str(cats[m])
                cats_label[m] = cats_label[m] + ' (%d)' % len(dfs[m].drop_duplicates('particle'))
        else:
            for m in range(len(cats_label)):
                cats_label[m] = cats_label[m] + ' (%d)' % len(dfs[m].drop_duplicates('particle'))

    else:
        dfs = [df]
        cats_label = ['MSD']
        colors = [(0, 0, 0, RGBA_alpha)]

    for i, df in enumerate(dfs):

        # Calculate individual msd
        im = tp.imsd(df, mpp=pixel_size, fps=frame_rate, max_lagtime=np.inf)
        #cut the msd curves and convert units to nm

        n = len(im.index) #for use in stand err calculation
        m = int(round(len(im.index)/divide_num))
        im = im.head(m)
        im = im*1e6

        if len(im) > 1:

            # """
            # ~~~~~~~~~~~Plot the mean MSD data and error bar~~~~~~~~~~~~~~
            # """
            imsd_mean = im.mean(axis=1)
            imsd_std = im.std(axis=1, ddof=0)

            #print(imsd_std)
            x = imsd_mean.index.to_numpy()
            y = imsd_mean.to_numpy()
            n_data_pts = np.sqrt(np.linspace(n-1, n-m, m))
            yerr = np.divide(imsd_std.to_numpy(), n_data_pts)

            # # """
            # # ~~~~~~~~~~~Plot the fit of the average~~~~~~~~~~~~~~
            # # """
            popt_log = fit_msd(x, y, space='log')
            fit_of_mean_msd = msd(x, popt_log[0], popt_log[1])
            ax.plot(x, fit_of_mean_msd, color=colors[i],
                    label=cats_label[i], linewidth=fitting_linewidth)
            ax.errorbar(x, fit_of_mean_msd, yerr=yerr, linestyle='None',
                marker='.', markersize=markersize,
                elinewidth=elinewidth, capsize=capsize, color=colors[i])

    # """
    # ~~~~~~~~~~~Set the label~~~~~~~~~~~~~~
    # """
    if set_format:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(labelsize=13, width=2, length=5)

        ax.set_xlabel(r'$\mathbf{Time (s)}$', fontsize=15)
        ax.set_ylabel(r'$\mathbf{MSD(nm^2)}$', fontsize=15)
