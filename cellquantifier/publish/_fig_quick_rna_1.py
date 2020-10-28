import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from cellquantifier.segm import clr_code_mask
from cellquantifier.plot.plotutil import format_ax
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_ubyte
from copy import deepcopy

def fig_quick_rna(blobs_df, int_df, im_arr, mask,
				  cls_div_mat=[[1,.1],[1,-.1]],
				  min_class_int=[.1,.1],
				  typ_arr=['type1','type2','type3','type4'],
				  typ_clr_arr=[(255,255,255),(0,255,0),\
							   (255,255,0),(255,0,0)]):

	"""

	Validation figure 1 for RNA expression analysis

	Pseudo code
	----------
	1. Build the figure
	2. Populate upper right panel with cell classifications
	3. Populate left panel with overlay image and cell classifications
	4. Merge blobs_df and int_df on label column
	5. Generate blox plots with copy number and peak intensity per cell

	"""

	#Ensure blobs_df only contains one frame
	blobs_df = blobs_df.loc[blobs_df['frame'] == 0]

	# """
	# ~~~~~~~~~~~Build figure (1)~~~~~~~~~~~~~~
	# """

	fig = plt.figure(figsize=(9,9))
	ax0 = plt.subplot2grid((9,9), (0, 0), rowspan=4, colspan=4)
	ax1 = plt.subplot2grid((9,9), (0, 5), rowspan=4, colspan=4)
	ax2 = plt.subplot2grid((9,9), (5, 0), rowspan=4, colspan=4)
	ax3 = plt.subplot2grid((9,9), (5, 5), rowspan=4, colspan=2)
	ax4 = plt.subplot2grid((9,9), (5, 7), rowspan=4, colspan=2)

	# """
	# ~~~~~~~~~~~Upper Right Panel (2)~~~~~~~~~~~~~~
	# """

	#Classify cells by partitioning mean intensity space
	x = np.linspace(0,1,100)
	tmp1, tmp2 = np.zeros_like(x), np.ones_like(x)

	m1,b1 = cls_div_mat[0]; y1 = m1*x + b1
	m2,b2 = cls_div_mat[1]; y2 = m2*x + b2

	ax1.plot(x, y1, x, y2, color='black', linewidth=5)
	ax1.fill_between(x, tmp1, y2, color='green')
	ax1.fill_between(x, tmp2, y1, color='white')
	ax1.fill_between(x, y1, y2, color='yellow')

	rect = Rectangle((0,0), min_class_int[0], min_class_int[1],\
					 color='red', fill=False)
	ax1.add_patch(rect)

	ax1.scatter(int_df['avg_ins_intensity'], \
			   int_df['avg_gluc_intensity'], \
			   color='blue', s=20, marker='s')


	format_ax(ax1, ax_is_box=False, xlabel=r'$I^{\beta}$',
			  ylabel=r'$I^{\alpha}$', label_fontsize=20,
			  xscale = [0, 1, 1, .1],
			  yscale = [0, 1, 1, .1])

	# """
	# ~~~~~~~~~~~Left Panel (3)~~~~~~~~~~~~~~
	# """

	mask_rgb = clr_code_mask(mask, int_df, typ_arr, typ_clr_arr)
	ax0.imshow(im_arr[0], alpha=0.5)
	ax0.imshow(mask_rgb, alpha=0.5)
	ax2.imshow(im_arr[1],alpha=0.5)
	ax2.imshow(mask_rgb, alpha=0.5)
	ax2.scatter(blobs_df['y'], blobs_df['x'], s=2, color='blue')

	# """
	# ~~~~~~~~~~~Merge Blobs/Intensity DataFrames (4)~~~~~~~~~~~~~~
	# """

	blobs_df = pd.merge(blobs_df, int_df, on="label")

	# """
	# ~~~~~~~~~~~Lower Right Panel 1 (5)~~~~~~~~~~~~~~
	# """

	count_df = blobs_df.groupby(['label', \
								 'cell_type']).size().reset_index(name="count")

	count_df_arr = [count_df.loc[count_df['cell_type'] == typ, 'count'] \
					for typ in typ_arr]

	bp1 = ax3.boxplot(count_df_arr, showfliers=False, patch_artist=True)
	format_ax(ax3, ax_is_box=False)
	ax3.set_ylabel(r'$\mathbf{Copy-Number}$')
	ax3.set_xticks([])

	# """
	# ~~~~~~~~~~~Lower Right Panel 2 (5)~~~~~~~~~~~~~~
	# """

	peak_df_arr = [blobs_df.loc[blobs_df['cell_type'] == typ, 'peak'] \
					for typ in typ_arr]

	bp2 = ax4.boxplot(peak_df_arr, showfliers=False, patch_artist=True)
	format_ax(ax4, ax_is_box=False)
	ax4.set_ylabel(r'$\mathbf{Peak-Intensity}$')
	ax4.set_xticks([])

	# """
	# ~~~~~~~~~~~Set box plot colors~~~~~~~~~~~~~~
	# """

	colors = np.array(typ_clr_arr)/255
	for bplot in (bp1, bp2):
		for patch, color in zip(bplot['boxes'], colors):
			patch.set_facecolor(color)

	plt.tight_layout()
