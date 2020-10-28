import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np

from distutils.util import strtobool
from cellquantifier.plot.plotutil import *


def add_strip_plot(ax,
				   df,
				   hist_col,
				   cat_col,
				   xlabels=None,
				   ylabel=None,
				   palette=None,
				   counts=False,
				   x_labelsize=10,
				   y_labelsize=10,
				   drop_duplicates=False):

	"""Generate strip plot

	Parameters
	----------

	Parameters
	----------
	ax : object
		matplotlib axis

	df : DataFrame
		DataFrame containing hist_col, cat_col

	hist_col : str
		Column of df that contains the data

	cat_col : str
		Column to use for categorical sorting

	palette : array
		seaborn color palette

	"""

	if drop_duplicates:
		df = df.drop_duplicates('particle')

	pal = sns.color_palette(palette)

	sns.stripplot(x=cat_col,
				   y=hist_col,
				   data=df,
				   palette=pal,
				   s=3,
				   alpha=.65,
				   ax=ax)

	# """
	# ~~~~~~~~~~~Add the mean bars~~~~~~~~~~~~~~
	# """

	median_width = 0.3
	i = 0
	labels = []

	x = ax.get_xticklabels()

	for tick, text in zip(ax.get_xticks(), x):

		if isinstance(text.get_text(), str):
			sample_name = bool(strtobool(text.get_text()))

		x = df[df[cat_col]==sample_name][hist_col]

		if xlabels:
			label = xlabels[i]
		else:
			label = text.get_text()

		if counts:
			label += " (" + str(len(x)) + ")"

		labels.append(label)

		i+=1
		mean = x.mean()
		ax.plot([tick-median_width/2, tick+median_width/2], [mean, mean],
				lw=3, color='black')

	ax.set_xticklabels(labels)


	# """
	# ~~~~~~~~~~~Formatting~~~~~~~~~~~~~~
	# """

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_linewidth(2)
	ax.spines['bottom'].set_linewidth(2)
	ax.tick_params(labelsize=x_labelsize, width=2, length=5, labelrotation=20.0, axis='x')

	ax.set_ylabel(r'$\mathbf{' + ylabel + '}$', fontsize=y_labelsize)
	ax.set_xlabel('')
	ax.set_xlim(-.5,1.5)
