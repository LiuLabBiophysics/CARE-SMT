import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
import scipy.stats as stats

def add_gmm(ax, blobs_df, cat_col,
            n_comp,
            hist_col,
            cv_type='full',
            RGBA_alpha=0.5):
    """
    Add histogram in matplotlib axis.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.

    blobs_df : DataFrame
		DataFrame containing 'particle', 'D' columns

    cat_col : str
		Column to use for categorical sorting

    Returns
    -------
    Add D histograms in the ax.

    Examples
	--------
    import matplotlib.pyplot as plt
    import pandas as pd
    from cellquantifier.plot.plotutil import add_D_hist, add_gmm
    path = 'cellquantifier/data/physDataMerged.csv'
    df = pd.read_csv(path, index_col=None, header=0)
    fig, ax = plt.subplots()
    add_D_hist(ax, df, 'exp_label',
                RGBA_alpha=0.5)
    add_gmm(ax, df, 'exp_label', n_comp=3, hist_col='D',
                RGBA_alpha=0.5)
    plt.show()

    """


    # """
    # ~~~~~~~~~~~Check if blobs_df is empty~~~~~~~~~~~~~~
    # """

    if blobs_df.empty:
    	return


    # """
    # ~~~~~~~~~~~Prepare the data, category, color~~~~~~~~~~~~~~
    # """

    cats = sorted(blobs_df[cat_col].unique())
    blobs_dfs = [blobs_df.loc[blobs_df[cat_col] == cat] for cat in cats]
    colors = plt.cm.jet(np.linspace(0,1,len(cats)))
    colors[:, 3] = RGBA_alpha

    for i, blobs_df in enumerate(blobs_dfs):

        this_df = blobs_df.drop_duplicates(subset='particle')[hist_col]

        f = np.ravel(this_df).astype(np.float)

        f = f.reshape(-1,1)
        g = mixture.GaussianMixture(n_components=n_comp,covariance_type=cv_type)
        g.fit(f)
        bic = g.bic(f)
        log_like = g.score(f)

        gmm_df = pd.DataFrame()
        gmm_df['weights'] = g.weights_
        gmm_df['means'] = g.means_
        gmm_df['covar'] = g.covariances_.reshape(1,n_comp)[0]
        gmm_df = gmm_df.sort_values(by='means')

        f_axis = f.copy().ravel()
        f_axis.sort()



        for j in range(n_comp):
        	label = r'$\mu$=' + str(round(gmm_df['means'].to_numpy()[j], 2))\
        			+ r' $\sigma$=' + str(round(np.sqrt(gmm_df['covar'].to_numpy()[j]), 2))
        	ax.plot(f_axis,gmm_df['weights'].to_numpy()[j]*stats.norm.pdf(\
        				 f_axis,gmm_df['means'].to_numpy()[j],\
        				 np.sqrt(gmm_df['covar'].to_numpy()[j])).ravel(),\
        				 c=colors[i], label=label)

        # ax.pie(gmm_df['weights'], autopct='%1.1f%%')

    # """
    # ~~~~~~~~~~~Set the label~~~~~~~~~~~~~~
    # """
    ax.legend()
