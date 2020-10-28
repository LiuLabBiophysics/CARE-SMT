import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import pearsonr, spearmanr
from cellquantifier.qmath import fit_linear

def add_corr_coeff(ax,
				   df,
				   col1,
				   col2,
				   corr_type='linear',
				   norm=False,
				   color='black',
				   text_pos=[.75,.8]):


	"""Add pearson correlation coefficient to axis

	Parameters
	----------

	Parameters
	----------
	ax : object
		matplotlib axis

	df : DataFrame
		DataFrame containing hist_col, cat_col

	col1,col2 : str
		Column names that contain the data

	norm : bool
		Transform data to arbitrary units

	"""

	x,y = df[col1], df[col2]

	if norm:
		x = x/np.abs(x).max()
		y = y/np.abs(y).max()

	if corr_type == 'linear':

		r, p = pearsonr(x,y)

	elif corr_type == 'nonlinear':

		r,p = spearmanr(x,y)

	ax.text(text_pos[0],
			text_pos[1],
			r'$\mathbf{\rho}$' + """: % .2f""" %(r),
			fontsize = 12,
			color = color,
			transform=ax.transAxes)
