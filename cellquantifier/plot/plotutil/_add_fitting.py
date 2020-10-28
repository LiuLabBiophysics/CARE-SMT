import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import pearsonr
from cellquantifier.qmath import fit_linear

def add_fitting(ax,
				df,
				col1,
				col2,
				norm=False,
				color='lime',
				label=None):

	"""Add linear fitting to axis

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

	slope, intercept, r, p = fit_linear(x,y)

	ax.plot(x, intercept + slope*x, color, label=label)
