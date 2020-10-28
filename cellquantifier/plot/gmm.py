import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker as tkr
import scipy.stats as stats
import pandas as pd
import itertools
import matplotlib as mpl

from matplotlib import rc
from sklearn import mixture
from sklearn import mixture
from scipy import linalg

def plot_gmm_selection(df,
					   cat_col,
					   hist_col,
					   max_comp=5):

	"""
	Fit a GaussianMixtureModel (GMM) to data iteratively and find best model

	Parameters
	----------
	df : DataFrame object
		The DataFrame containing cat_col and hist_col columns
	cat_col : str
		Column to use for categorical sorting
	hist_col : str
		Column that contains the actual data
	max_comp : int, optional
		The maximum number of components to test for the GMM

	"""

	cats = df[cat_col].unique()
	lowest_bic = np.infty
	bic = []
	log_like = []
	n_components_range = range(1, max_comp)

	for cat in cats:
		this_df = df.loc[df[cat_col] == cat]
		for n_components in n_components_range:
			# Fit a Gaussian mixture with EM
			f = np.ravel(this_df[hist_col]).astype(np.float)
			f = f.reshape(-1,1)
			gmm = mixture.GaussianMixture(n_components=n_components,
										  covariance_type='full')
			gmm.fit(f)
			# bic.append(2 * gmm.score(f) * len(f) +\
            #     n_components * np.log(len(f)))
			bic.append(gmm.bic(f))
			log_like.append(gmm.score(f))
			if bic[-1] < lowest_bic:
				lowest_bic = bic[-1]
				best_gmm = gmm

	bic = np.array(bic)

	color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
								  'darkorange'])
	clf = best_gmm
	bars = []
	bars2 = []

	fix, ax = plt.subplots(1,2)

	for i, (cat, color) in enumerate(zip(cats, color_iter)):
		xpos = np.array(n_components_range) + .2 * (i - 2)
		bars.append(ax[1].bar(xpos, bic[i * len(n_components_range):
									  (i + 1) * len(n_components_range)],
							width=.2, color=color))
		bars2.append(ax[0].bar(xpos, log_like[i * len(n_components_range):
									  (i + 1) * len(n_components_range)],
							width=.2, color=color))

	ax[1].set_xticks(n_components_range)
	ax[1].set_ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	ax[1].set_title('BIC score per model')
	ax[1].set_xlabel('Number of components')
	ax[1].legend([b[0] for b in bars], cats)

	ax[0].set_title('Log likelihood')
	ax[0].set_xticks(n_components_range)
	ax[0].set_ylim([0, min(log_like)])
	ax[0].set_xlabel('Number of components')
	ax[0].legend([b[0] for b in bars], cats)



	plt.tight_layout()
	plt.show()


def plot_gmm(df,
	n_comp,
	cat_col,
	hist_col,
	cv_type='full',
	nbins=100,
	pltshow=True):

	"""
	Fit a GaussianMixtureModel (GMM) to data

	Parameters
	----------
	df : DataFrame object
		The DataFrame containing cat_col and hist_col columns
	n_comp : int
		Number of gaussian components in the GMM
	cat_col : str
		Column to use for categorical sorting
	hist_col : int
		Column containing the data to be fit
	cv_type: str, optional
		The type of covariance to use
	pltshow : bool, optional
		Whether to show the figure or just save to disk


	Examples
	--------

	>>> import pandas as pd
	>>> from cellquantifier.plot import plot_gmm
	>>> path = 'cellquantifier/data/test_physDataMerged.csv'
	>>> df = pd.read_csv(path, index_col=None, header=0)
	>>> df = df.drop_duplicates('D')
	>>> plot_gmm(df, 5, 'exp_label', 'D')

	"""

	cats = df[cat_col].unique()
	fig,ax = plt.subplots(len(cats),len(cats))
	colors = plt.cm.jet(np.linspace(0,1,n_comp))

	for i, cat in enumerate(cats):

		this_df = df.loc[df[cat_col] == cat]
		this_df = this_df[hist_col]

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

		ax[0,i].hist(f, bins=nbins, histtype='bar', density=True, ec='red', alpha=0.5)
		ax[0,i].set_title(cat)
		ax[1,i].pie(gmm_df['weights'], colors=colors, autopct='%1.1f%%')

		for j in range(n_comp):
			label = r'$\mu$=' + str(round(gmm_df['means'].to_numpy()[j], 2))\
					+ r' $\sigma$=' + str(round(np.sqrt(gmm_df['covar'].to_numpy()[j]), 2))
			ax[0,i].plot(f_axis,gmm_df['weights'].to_numpy()[j]*stats.norm.pdf(\
						 f_axis,gmm_df['means'].to_numpy()[j],\
						 np.sqrt(gmm_df['covar'].to_numpy()[j])).ravel(),\
						 c=colors[j], label=label)

		ax[0,i].legend(fontsize=8)

		textstr = '\n'.join((

			r'$\hat{L}: %.2f$' % (log_like),
			r'$BIC: %.2f$' % (bic)))


		props = dict(boxstyle='round', facecolor='wheat', alpha=0.0)
		ax[0,i].text(.7, .8, textstr, transform=ax[0,i].transAxes,  \
					horizontalalignment='left', verticalalignment='top',\
					fontsize=8, color='black', bbox=props)

	plt.rcParams['agg.path.chunksize'] = 10000
	plt.grid()
	plt.tight_layout()

	if pltshow:
		plt.show()

	return g

def plot_gmm_temp(df,
	n_comp1,
	n_comp2,
	cat_col1,
	cat_col2,
	cv_type='full',
	nbins=100,
	pltshow=True):

	"""
	Fit a GaussianMixtureModel (GMM) to data

	Parameters
	----------
	df : DataFrame object
		The DataFrame containing cat_col and hist_col columns
	n_comp : int
		Number of gaussian components in the GMM
	cat_col1 : str
		Column to use for categorical sorting
	cat_col2 : str
		Column to use for categorical sorting
	cv_type: str, optional
		The type of covariance to use
	pltshow : bool, optional
		Whether to show the figure or just save to disk


	Examples
	--------

	>>> import pandas as pd
	>>> from cellquantifier.plot import plot_gmm
	>>> path = 'cellquantifier/data/test_physDataMerged.csv'
	>>> df = pd.read_csv(path, index_col=None, header=0)
	>>> df = df.drop_duplicates('D')
	>>> plot_gmm(df, 5, 'exp_label', 'D')

	"""

	cats1 = sorted(df[cat_col1].unique())
	cats2 = sorted(df[cat_col2].unique())
	hist_cols=['D', 'alpha']
	n_comp=[n_comp1, n_comp2]
	line_types = ['-.','--', '-']
	dist_labels=['Far', 'Near']
	fig,ax = plt.subplots(len(cats1),len(cats2))
	# colors = plt.cm.jet(np.linspace(0,1,n_comp))
	hist_colors=['red','blue']

	for i, cat1 in enumerate(cats1):

		df1 = df.loc[df[cat_col1] == cat1]
		ax[i,0].set_ylabel(cat1)

		for j, hist_col in enumerate(hist_cols):

			df2 = df1[[hist_col, cat_col2]]

			for k, cat2 in enumerate(cats2):

				this_df = df2.loc[df2[cat_col2] == cat2]

				this_df = this_df[hist_col]

				f = np.ravel(this_df).astype(np.float)

				f = f.reshape(-1,1)
				g = mixture.GaussianMixture(n_components=n_comp[j],covariance_type=cv_type)
				g.fit(f)
				bic = g.bic(f)
				log_like = g.score(f)

				gmm_df = pd.DataFrame()
				gmm_df['weights'] = g.weights_
				gmm_df['means'] = g.means_
				gmm_df['covar'] = g.covariances_.reshape(1,n_comp[j])[0]
				gmm_df = gmm_df.sort_values(by='means')

				f_axis = f.copy().ravel()
				f_axis.sort()

				ax[i,j].hist(f, bins=nbins, histtype='bar', density=True, ec=hist_colors[k], alpha=0.4)

				for m in range(n_comp[j]):
					label = r'$\mu$=' + str(round(gmm_df['means'].to_numpy()[m], 2))\
							+ r' $\sigma$=' + str(round(np.sqrt(gmm_df['covar'].to_numpy()[m]), 2))
					ax[i,j].plot(f_axis,gmm_df['weights'].to_numpy()[m]*stats.norm.pdf(\
								 f_axis,gmm_df['means'].to_numpy()[m],\
								 np.sqrt(gmm_df['covar'].to_numpy()[m])).ravel(),\
								 c=hist_colors[k], label=label, linestyle=line_types[m])

					props = dict(boxstyle='round', facecolor='wheat', alpha=0.0)

					ax[i,j].text(.6, .8+.1*(k), dist_labels[k], transform=ax[i,j].transAxes,  \
								horizontalalignment='left', verticalalignment='top',\
								fontsize=12, color=hist_colors[k], bbox=props)


				ax[i,j].legend(fontsize=8)
				ax[i,j].set_title(hist_col)

				# textstr = '\n'.join((
				#
				# 	r'$\hat{L}: %.2f$' % (log_like),
				# 	r'$BIC: %.2f$' % (bic)))
				#
				#
				# props = dict(boxstyle='round', facecolor='wheat', alpha=0.0)
				# ax[i,j].text(.7, .8, textstr, transform=ax[i,j].transAxes,  \
				# 			horizontalalignment='left', verticalalignment='top',\
				# 			fontsize=8, color='black', bbox=props)

	plt.rcParams['agg.path.chunksize'] = 10000
	plt.tight_layout()

	if pltshow:
		plt.show()

	return g
