import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp

from scipy.stats import sem
from ...qmath import msd, fit_msd

def add_53bp1_diffusion(ax, df,
					exp_col='exp_label',
					cell_col='cell_num',
					cycle_col='cycle_num',
					dt_per_cycle=2,
					start=None,
					RGBA_alpha=0.5,
					fitting_linewidth=3,
					elinewidth=None,
					markersize=None,
					capsize=2,
					set_format=True):

	"""
	Add mean D curve in matplotlib axis.
	For use with cycled imaging only

	Parameters
	----------
	ax : object
		matplotlib axis to annotate ellipse.

	df : DataFrame
		DataFrame containing 'cell_num', 'particle', 'exp_label', 'cycle_num'
		columns

	cat_col : None or str
		Column to use for categorical sorting

	"""


	# """
	# ~~~~~~~~~~~Check if df is empty~~~~~~~~~~~~~~
	# """

	if df.empty:
		return

	# """
	# ~~~~~~~~~~~Divide the data by exp_label~~~~~~~~~~~~~~
	# """

	exps = df[exp_col].unique()
	# colors = plt.cm.coolwarm(np.linspace(0,1,len(exps)))
	colors = ['blue', 'red']
	d_coeff = [[] for exp in exps]
	exp_dfs = [df.loc[df[exp_col] == exp] for exp in exps]

	for i, exp_df in enumerate(exp_dfs):

		# """
		# ~~~~~~~~~~~Divide the data by cell~~~~~~~~~~~~~~
		# """

		cells = exp_df[cell_col].unique()
		cell_dfs = [exp_df.loc[exp_df[cell_col] == cell] for cell in cells]

		for j, cell_df in enumerate(cell_dfs):

		# """
		# ~~~~~~~~~~~Divide the data by cycle~~~~~~~~~~~~~~
		# """

			d_coeff[i].append([])
			cycles = sorted(cell_df[cycle_col].unique())
			cycle_dfs = [cell_df.loc[cell_df[cycle_col] == cycle]\
									for cycle in cycles]

			for k, cycle_df in enumerate(cycle_dfs):
				D = cycle_df.drop_duplicates('particle')['D']
				mean_D = D.mean()
				d_coeff[i][j].append(mean_D)

	# """
	# ~~~~~~~~~~Compute the mean and add to axis~~~~~~~~~~~~~~
	# """

		mean_mean_D = np.mean(d_coeff[i], axis=0)
		yerr = sem(d_coeff[i], axis=0)

		if start:
			trunc = cycles[start-1:] #truncate cycle array
			delta = trunc[0]
			shifted = np.array([cycle-delta for cycle in trunc])
			shifted = shifted*dt_per_cycle
			mean_mean_D = mean_mean_D[start-1:]
			yerr = yerr[start-1:]

			ax.errorbar(shifted, mean_mean_D, yerr=yerr,
						color=colors[i], linestyle='-', label=exps[i])

		else:
			cycles = np.array(cycles)
			cycles = cycles*dt_per_cycle
			ax.errorbar(shifted, mean_mean_D, yerr=yerr,
						color=colors[i], linestyle='-', label=exps[i])


	# """
	# ~~~~~~~~~~~Set the label~~~~~~~~~~~~~~
	# """

	ax.axvspan(-dt_per_cycle, -.5*dt_per_cycle, alpha=0.3, color='gray')
	ax.axvspan(-.5*dt_per_cycle, 0, alpha=0.3, color='red')
	ax.set_xlim(-dt_per_cycle, (max(cycles))*dt_per_cycle)

	if set_format:
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_linewidth(2)
		ax.spines['bottom'].set_linewidth(2)
		ax.tick_params(labelsize=13, width=2, length=5)
		ax.set_xlabel(r'$\mathbf{Time (min)}$', fontsize=15)
		ax.set_ylabel(r'$\mathbf{D (nm^{2}/s)}$', fontsize=15)
		ax.legend()
