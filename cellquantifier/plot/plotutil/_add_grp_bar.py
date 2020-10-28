from ._format_ax import format_ax
from ...phys import *
from ...util import *
import numpy as np

def add_grp_bar(ax, df,
				cat_cols,
				colors,
				grp_space=1.0,
				bar_width=.35,
				alpha=1.0,
				err_col='sem',
				neg_err=False,
				stack_col=None,
				btm_clr=None):

	"""
	Add a grouped bar chart to the axis. See
	https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html

	A single layer of grouped bar charts is added to the axis for 2D data.
	For 3D data, where the third dimension is binary, two layers of grouped
	bar charts are added to the axis, one stacked on top of the other


	Pseudo code
	----------
	1. If stack column is specified, divide the data on stack col
	2. Call add_layer() for each sub-dataframe

	Parameters
	----------
	ax : object
		matplotlib axis.

	df : DataFrame
		DataFrame that contains cat_cols

	cat_cols : list
		a list of columns, each of which is a category to divide the data on

	colors : list
		a list of colors

	neg_err : bool, optional
		whether or not to show the negative error in error bars, False by defualt

	grp_space : float, optional
		spacing between grouped bars in axis units

	bar_width : float, optional
		width of each bar

	alpha : list, optional
		transparency of the bars

	err_col : str, optional
		name of the DataFrame column containing error values e.g. 'sem' or 'std'

	stack_col : str, optional
		a column to divide the DataFrame on, each sub-dataframe defining a layer
		of a layered bar chart

	btm_clr : list, optional
		an additional list of colors to use strictly for the bottom layers
		of bars in a stacked bar scenario. applies only for 3-dimensional data.

	Example
	----------
	"""

	shape = tuple([df[col].nunique() for col in cat_cols])

	# """
	# ~~~~~~~~~~~Check input~~~~~~~~~~~~
	# """

	if not err_col in df.columns:
		print('Error: %s not found in DataFrame' % str(err_col))
		return

	if not colors:
		print('Error: please specify a list of colors')


	# """
	# ~~~~~~~~~~~Define add_layer()~~~~~~~~~~~
	# """


	def add_layer(ax, ind_arr, mean_arr, err_arr,
				  colors=None, bottom=None):

		if not bottom:
			bottom = list(np.zeros_like(np.array(mean_arr)))

		for x, b_arr, h_arr, s_arr in zip(ind_arr, bottom, mean_arr, err_arr):
			ax.bar(x, h_arr, bar_width, alpha=alpha, \
				   bottom=b_arr, color=colors, edgecolor='black')
			tmp = np.zeros_like(s_arr)
			err_pos = np.array(b_arr) + np.array(h_arr)
			ax.errorbar(x, err_pos, yerr=[tmp, s_arr],
						color='black', capsize=3, ls='none')

	# """
	# ~~~~~~~~~~~Convert dataframes to nest_df arrays recursively~~~~~~~~~~~~
	# """

	if stack_col:
		df_arr = nest_df(df, cat_cols=[stack_col])
		ind_arr = get_bar_positions(shape, bar_width=bar_width,
									grp_space=grp_space)
		# """
		# ~~~~~~~~Bottom~~~~~~~~~~~
		# """

		if not btm_clr:
			btm_clr = colors

		btm_mean_arr = nest_df(df_arr[0], cat_cols, data_col='mean')
		btm_err_arr = nest_df(df_arr[0], cat_cols, data_col=err_col)
		add_layer(ax, ind_arr, btm_mean_arr, btm_err_arr,
				  colors=btm_clr, bottom=None)

		# """
		# ~~~~~~~~~Top~~~~~~~~~~~~~~~
		# """

		top_mean_arr = nest_df(df_arr[1], cat_cols, data_col='mean')
		top_err_arr = nest_df(df_arr[1], cat_cols, data_col=err_col)
		add_layer(ax, ind_arr, top_mean_arr, top_err_arr,
			      colors=colors, bottom=btm_mean_arr)

		mean_arr = np.array(btm_mean_arr) + np.array(top_mean_arr)
		return (ind_arr, mean_arr, top_err_arr)

	else:
		mean_arr = nest_df(df, cat_cols, data_col='mean')
		err_arr = nest_df(df, cat_cols, data_col=err_col)
		ind_arr = get_bar_positions(shape, bar_width=bar_width,
									grp_space=grp_space)

		add_layer(ax, ind_arr, mean_arr, err_arr, colors=colors)

		return (ind_arr, mean_arr, err_arr)


def get_bar_positions(shape, bar_width, grp_space):

	pos_arr = []
	for x in range(1, shape[0]+1):
		x = x*grp_space
		delta = bar_width*(shape[1]-1)/2
		tmp = list(np.linspace(x-delta, x+delta, shape[1]))
		if len(shape) == 3:
			tmp = [[y, y] for y in tmp]
		pos_arr.append(tmp)

	return pos_arr
