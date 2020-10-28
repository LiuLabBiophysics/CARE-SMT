import numpy as np
import trackpy as tp
import matplotlib.pyplot as plt
import seaborn as sns
from ...qmath import msd, fit_msd

def add_mean_msd2(ax, df,

                pixel_size,
                frame_rate,
                divide_num,

                cat_col=None,
                cat_order=None,
                color_order=None,
                RGBA_alpha=1,

                fitting_linewidth=3,
                elinewidth=None,
                markersize=None,
                capsize=2,
                ):
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
    """


    # """
    # ~~~~Check if df is empty~~~~
    # """

    if df.empty:
    	return

    # """
    # ~~~~Prepare the data, category, color~~~~
    # """
    if cat_col:
        # cats
        if cat_order:
            cats = cat_order
        else:
            cats = sorted(df[cat_col].unique())

        # dfs
        dfs = [df.loc[df[cat_col] == cat] for cat in cats]

        # cats_label
        cats_label = cats.copy()
        for i in range(len(cats_label)):
            cats_label[i] = cats_label[i] + ' (%d)' \
                            % len(dfs[i].drop_duplicates('particle'))

        # colors
        if color_order:
            palette = np.array(color_order)
            colors = np.zeros((palette.shape[0], 4))
            colors[:, 0:3] = palette[:, 0:3]
            colors[:, 3] = RGBA_alpha
        else:
            palette = np.array(sns.color_palette('muted'))
            colors = np.zeros((palette.shape[0], 4))
            colors[:, 0:3] = palette[:, 0:3]
            colors[:, 3] = RGBA_alpha
    else:
        dfs = [df]
        cats_label = ['MSD']
        colors = [(0, 0, 0, RGBA_alpha)]

    # """
    # ~~~~Prepare the data, category, color~~~~
    # """
    for i, df in enumerate(dfs):
        im = tp.imsd(df,
                    mpp=pixel_size,
                    fps=frame_rate,
                    max_lagtime=np.inf,
                    )

        n = len(im.index) #for use in stand err calculation
        m = int(round(len(im.index)/divide_num))
        im = im.head(m)
        im = im*1e6

        if len(im) > 1:

            # """
            # ~~~~~~Plot the mean MSD data and error bar~~~~
            # """
            imsd_mean = im.mean(axis=1)
            imsd_std = im.std(axis=1, ddof=0)
            x = imsd_mean.index.to_numpy()
            y = imsd_mean.to_numpy()
            n_data_pts = np.sqrt(np.linspace(n-1, n-m, m))
            yerr = np.divide(imsd_std.to_numpy(), n_data_pts)

            # """
            # ~~~~~~~~~~~Plot the fit of the average~~~~~~~~~~~~~~
            # """
            popt_log = fit_msd(x, y, space='log')
            fit_of_mean_msd = msd(x, popt_log[0], popt_log[1])
            ax.plot(x, fit_of_mean_msd, '-',
                    color=colors[i],
                    label=cats_label[i],
                    linewidth=fitting_linewidth,
                    )
            ax.errorbar(x, fit_of_mean_msd, yerr=yerr,
                    linestyle='None',
                    marker='.',
                    markersize=markersize,
                    elinewidth=elinewidth,
                    capsize=capsize,
                    color=colors[i],
                    )
            ax.legend()
