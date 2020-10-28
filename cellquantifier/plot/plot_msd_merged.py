import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import trackpy as tp
import pandas as pd
from datetime import date
import matplotlib as mpl
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.weight'] = 'bold'

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from ..qmath import msd, fit_msd, t_test
from skimage.io import imsave
from ..plot.plotutil import plt2array


def plot_msd_merged(blobs_df,
			 cat_col,
			 output_path,
			 root_name,
			 pixel_size,
			 frame_rate,
			 divide_num,
			 pltshow=True):

	fig, ax = plt.subplots(1,3, figsize=(18, 6))
	cats = blobs_df[cat_col].unique()
	blobs_dfs = [blobs_df.loc[blobs_df[cat_col] == cat] for cat in cats]
	colors = plt.cm.jet(np.linspace(0,1,len(cats)))

	#
	# ~~~~~~~~~~~Run a t-test~~~~~~~~~~~~~~
	# """

	if len(cats) == 2:
		x = blobs_df.drop_duplicates('particle')[['D', 'alpha', cat_col]]
		D_stats = t_test(x.loc[x[cat_col] == cats[0]]['D'], \
						   x.loc[x[cat_col] == cats[1]]['D'])
		alpha_stats = t_test(x.loc[x[cat_col] == cats[0]]['alpha'], \
						   x.loc[x[cat_col] == cats[1]]['alpha'])
	#
	# ~~~~~~~~~~~Check if blobs_df is empty~~~~~~~~~~~~~~
	# """

	for i, blobs_df in enumerate(blobs_dfs):

		if blobs_df.empty:
			return

		# Calculate individual msd
		im = tp.imsd(blobs_df, mpp=pixel_size, fps=frame_rate, max_lagtime=np.inf)

		#Get the diffusion coefficient for each individual particle
		D_ind = blobs_df.drop_duplicates('particle')['D'].mean()

		# """
		# ~~~~~~~~~~~Get the avg/stdev for D,alpha of each particle~~~~~~~~~~~~~~
		# """

		if len(im) > 1:

			d_avg = (blobs_df.drop_duplicates(subset='particle')['D']).mean() #average D value
			d_std = (blobs_df.drop_duplicates(subset='particle')['D']).std() #stdev of D value

			#avg/stdev of alpha
			alpha_avg = (blobs_df.drop_duplicates(subset='particle')['alpha']).mean() #average alpha value
			alpha_std = (blobs_df.drop_duplicates(subset='particle')['alpha']).std() #stdev of alpha value

	# """
	# ~~~~~~~~~~~Get the mean MSD curve and its standard dev~~~~~~~~~~~~~~
	# """

		#cut the msd curves and convert units to nm
		n = int(round(len(im.index)/divide_num))
		im = im.head(n)
		im = im*1e6

		imsd_mean = im.mean(axis=1)
		imsd_std = im.std(axis=1, ddof=0)

		x = imsd_mean.index
		y = imsd_mean.to_numpy()
		yerr = imsd_std.to_numpy()

		t = imsd_mean.index.to_numpy()
		popt_log = fit_msd(t,y, space='log') #fit the average msd curve in log space
		popt_lin = fit_msd(t,y, space='linear') #fit the average msd curve in linear space

	# """
	# ~~~~~~~~~~~Plot the fit of the average and the average of fits~~~~~~~~~~~~~~
	# """

		fit_of_avg = msd(t, popt_log[0], popt_log[1])
		avg_of_fits = msd(t, d_avg, alpha_avg)

		ax[0].errorbar(x, avg_of_fits, yerr=yerr, linestyle='None',
				marker='o', color=(colors[i][0], colors[i][1], colors[i][2], 0.5))

		ax[0].plot(t, avg_of_fits, '-', color=(colors[i][0], colors[i][1], colors[i][2], 0.5),
				   linewidth=4, markersize=12, label=cats[i])


		ax[0].set_xlabel(r'$\tau (\mathbf{s})$')
		ax[0].set_ylabel(r'$\langle \Delta r^2 \rangle$ [$nm^2$]')
		ax[0].legend()

		# """
		# ~~~~~~~~~~~Add D value histogram~~~~~~~~~~~~~~
		# """

		ax[1].hist(blobs_df.drop_duplicates(subset='particle')['D'].to_numpy(),
					bins=30, color=(colors[i][0], colors[i][1], colors[i][2], 0.5), label=cats[i], normed=True)

		textstr = '\n'.join((

			r'$t-value: %.2f$' % (D_stats[0]),
			r'$p-value: %.2f$' % (D_stats[1])))


		props = dict(boxstyle='round', facecolor='wheat', alpha=0.0)
		ax[1].text(.6, .8, textstr, transform=ax[1].transAxes,  horizontalalignment='left', verticalalignment='top', fontsize=12, color='black', bbox=props)

		ax[1].legend(loc='upper right')
		ax[1].set_ylabel('Frequency')
		ax[1].set_xlabel('D')

		# """
		# ~~~~~~~~~~~Add alpha value histogram~~~~~~~~~~~~~~
		# """

		ax[2].hist(blobs_df.drop_duplicates(subset='particle')['alpha'].to_numpy(),
					bins=30, color=(colors[i][0], colors[i][1], colors[i][2], 0.5), label=cats[i], normed=True)

		textstr = '\n'.join((

			r'$t-value: %.2f$' % (alpha_stats[0]),
			r'$p-value: %.2f$' % (alpha_stats[1])))


		props = dict(boxstyle='round', facecolor='wheat', alpha=0.0)
		ax[2].text(.6, .8, textstr, transform=ax[2].transAxes,  horizontalalignment='left', verticalalignment='top', fontsize=12, color='black', bbox=props)


		ax[2].legend(loc='upper right')
		ax[2].set_ylabel('Frequency')
		ax[2].set_xlabel('alpha')

	plt.tight_layout()

	# """
	# ~~~~~~~~~~~Save the plot as pdf, and open the pdf in browser~~~~~~~~~~~~~~
	# """
	start_ind = root_name.find('_')
	end_ind = root_name.find('_', start_ind+1)
	today = str(date.today().strftime("%y%m%d"))
	fig.savefig(output_path + today + root_name[start_ind:end_ind] + '-mergedResults.pdf')
	if pltshow:
		plt.show()
