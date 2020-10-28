import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp

from ...qmath import fit_spotcount
from ...qmath import spot_count as sc
from scipy.stats import sem
from ...qmath import msd, fit_msd

def add_53bp1_count(ax, df,
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
	Add mean spot count curve in matplotlib axis.
	For use with cycled imaging only
	The spot counts are obtained from df.

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
	spot_counts = [[] for exp in exps]
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

			spot_counts[i].append([])
			cycles = sorted(cell_df[cycle_col].unique())
			cycle_dfs = [cell_df.loc[cell_df[cycle_col] == cycle]\
									for cycle in cycles]

			for k, cycle_df in enumerate(cycle_dfs):
				spot_count = cycle_df['particle'].nunique()
				spot_counts[i][j].append(spot_count)

	# """
	# ~~~~~~~~~~Compute the mean and add to axis~~~~~~~~~~~~~~
	# """

		yerr = sem(spot_counts[i], axis=0)
		sc_mean = np.mean(spot_counts[i], axis=0)

		if start:
			trunc = cycles[start-1:] #truncate cycle array
			delta = trunc[0]
			shifted = np.array([cycle-delta for cycle in trunc])
			shifted = shifted*dt_per_cycle
			sc_mean = sc_mean[start-1:]
			popt = fit_spotcount(shifted, sc_mean)
			sc_mean_fit = sc(shifted, *popt)
			yerr = yerr[start-1:]

			tau_str = r': $\mathbf{\tau = %s}$' % round(popt[1], 2)
			ax.plot(shifted, sc_mean_fit, color=colors[i], label=exps[i] + tau_str)
			ax.errorbar(shifted, sc_mean, yerr=yerr, color=colors[i], linestyle='--')

		else:
			cycles = np.array(cycles)
			cycles = cycles*dt_per_cycle
			popt = fit_spotcount(cycles, sc_mean)
			sc_mean_fit = sc(cycles, *popt)

			tau_str = r': $\mathbf{\tau = %s}$' % round(popt[1], 2)
			ax.plot(cycles, sc_mean, color=colors[i], linestyle='--')
			ax.plot(cycles, sc_mean_fit, color=colors[i], label=exps[i] + tau_str)
			ax.errorbar(shifted, sc_mean, yerr=yerr, color=colors[i], linestyle='--')


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
		ax.set_ylabel(r'$\mathbf{Spot Count}$', fontsize=15)
		ax.legend()
