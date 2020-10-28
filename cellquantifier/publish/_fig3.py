import matplotlib.pyplot as plt
import pandas as pd

from seaborn import color_palette
from copy import deepcopy
from matplotlib.gridspec import GridSpec
from skimage.io import imread
from skimage.color import gray2rgb

from ..qmath import interpolate_lin
from ..util import bin_df
from ..plot.plotutil import *
from ..segm import get_thres_mask, get_dist2boundary_mask



def plot_fig_3(df,
			   bp1_thres=10,
			   nbins=10,
			   hole_size=1,
			   pixel_size=.1083,
			   frame_rate=3.33,
			   divide_num=5,
			   sub_df_path=None,
			   dutp_path=None,
			   bp1_path=None):

	"""
	Construct Figure 3

	Parameters
	----------

	df : DataFrame
	DataFrame containing 'particle', 'alpha' columns

	pixel_size : float

	frame_rate : float

	divide_num : float

	Example
	--------
	import pandas as pd
	from cellquantifier.publish import plot_fig_2
	from cellquantifier.phys.physutil import add_avg_dist
	from cellquantifier.plot.plotutil import *

	df = pd.read_csv('cellquantifier/data/physDataMerged.csv')
	df = add_avg_dist(df)

	boundary_sorter = [-20, 0]
	bp1_sorter = [-50, 10]

	df['sort_flag_boundary'] = df['avg_dist_bound'].between(boundary_sorter[0], \
														boundary_sorter[1],
														inclusive=True)

	df['sort_flag_53bp1'] = df['avg_dist_53bp1'].between(bp1_sorter[0], \
													 bp1_sorter[1],
													 inclusive=True)
	plot_fig_2(df)
	"""

	fig = plt.figure(figsize=(7,9))

	# """
	# ~~~~~~~~~~~Initialize Grid~~~~~~~~~~~~~~
	# """

	gs1 = GridSpec(6, 6)
	gs2 = GridSpec(4, 4)
	gs1.update(left=0.25, right=0.98, bottom=.15, top=.5, wspace=40, hspace=30)
	gs2.update(left=0.25, right=0.98, bottom=.55, top=.95, wspace=.1, hspace=30)

	ax1 = plt.subplot(gs1[:6, :4])
	ax2 = plt.subplot(gs1[:3, 4:])
	ax3 = plt.subplot(gs1[3:, 4:])

	ax4 = plt.subplot(gs2[:, :2])
	ax5 = plt.subplot(gs2[:, 2:])

	df_cpy = deepcopy(df)

	# """
	# ~~~~~~~~~~~BLM Data~~~~~~~~~~~~~~
	# """

	blm_df = df.loc[df['exp_label'] == 'BLM']
	blm_df = blm_df.loc[blm_df['avg_dist_53bp1'] < bp1_thres]

	blm_df_bincenters, blm_df_binned = bin_df(blm_df,
											  'avg_dist_53bp1',
											  nbins=nbins)

	blm_df_binned = blm_df_binned.groupby(['category'])
	D_blm = blm_df_binned['D'].mean().to_numpy()
	alpha_blm = blm_df_binned['alpha'].mean().to_numpy()

	r_cont_blm, D_blm = interpolate_lin(blm_df_bincenters,
										D_blm,
										pad_size=hole_size)

	r_cont_blm, alpha_blm = interpolate_lin(blm_df_bincenters,
											alpha_blm,
											pad_size=hole_size)

	# """
	# ~~~~~~~~~~~BLM MSD Curve~~~~~~~~~~~~~~
	# """

	blm_df_cpy = df_cpy.loc[df_cpy['exp_label'] == 'BLM']
	add_mean_msd(ax1,
			 blm_df_cpy,
			 pixel_size,
			 frame_rate,
			 divide_num,
			 'sort_flag_53bp1')
	ax1.set_title(r'$\mathbf{BLM}$', fontsize=18)

	# """
	# ~~~~~~~~~~~BLM Strip Plots~~~~~~~~~~~~~~
	# """

	palette = color_palette("coolwarm", 7)
	palette = [palette[0], palette[-1]]

	add_strip_plot(ax2,
			   blm_df_cpy,
			   'D',
			   'sort_flag_53bp1',
			   xlabels=['Far', 'Near'],
			   ylabel=r'\mathbf{D (nm^{2}/s)}',
			   palette=palette,
			   x_labelsize=12,
			   y_labelsize=14,
			   drop_duplicates=True)

	add_t_test(ax2,
		   blm_df_cpy,
		   cat_col='sort_flag_53bp1',
		   hist_col='D',
		   text_pos=[0.9, 0.9])

	add_strip_plot(ax3,
				blm_df_cpy,
				'alpha',
				'sort_flag_53bp1',
				xlabels=['Far', 'Near'],
				ylabel=r'\mathbf{\alpha}',
				palette=palette,
				x_labelsize=12,
				y_labelsize=14,
				drop_duplicates=True)


	add_t_test(ax3,
		   blm_df_cpy,
		   cat_col='sort_flag_53bp1',
		   hist_col='alpha',
		   text_pos=[0.9, 0.9])

	# """
	# ~~~~~~~~~~~Data Selection Figure~~~~~~~~~~~~~~
	# """

	df = pd.read_csv(sub_df_path)
	im_dutp = imread(dutp_path)[0]
	im_bp1 = imread(bp1_path)[0]

	im_zeros = np.zeros_like(im_dutp)
	im_dutp = np.dstack((im_zeros, im_dutp, im_zeros))
	im_bp1 = np.dstack((im_bp1, im_zeros, im_zeros))

	out = im_dutp + im_bp1
	out1 = out[255:275, 227:247]
	out2 = out[210:230, 188:208]
	ax4.imshow(out1)
	ax5.imshow(out2)

	ax4.set_xticks([])
	ax4.set_yticks([])
	ax5.set_xticks([])
	ax5.set_yticks([])

	for spine in ax4.spines.values():
		spine.set_edgecolor(palette[-1])
		spine.set_linewidth(5)
	for spine in ax5.spines.values():
		spine.set_edgecolor(palette[0])
		spine.set_linewidth(5)

	add_scalebar(ax4, .1084, sb_color=(1,1,1), sb_pos='lower left', fontsize=12)
	ax4.text(.02,
		    .9,
		    'dUTP',
		    fontsize=18,
		    color='green',
		    transform=ax4.transAxes)

	ax4.text(.02,
		    .8,
		    '53BP1',
		    fontsize=18,
		    color='red',
		    transform=ax4.transAxes)

	plt.tight_layout()
	# plt.savefig('/home/cwseitz/Desktop/fig3.png', dpi=1200)
	plt.show()
