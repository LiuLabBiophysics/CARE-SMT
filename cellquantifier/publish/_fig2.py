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

def plot_fig_2(df,
			   df_path,
			   im_path,
			   segm_path,
			   bound_thres=20,
			   hole_size=10,
			   nbins=12,
			   pixel_size=.1083,
			   frame_rate=3.33,
			   divide_num=5):

	"""
	Construct Figure 2

	Parameters
	----------

	df : DataFrame
		DataFrame containing 'particle', 'alpha' columns

	df_path: str,
		Path to csv file containing single cell dataframe for subfigure A

	im_path: str,
		Path to tif file containing single cell image for subfigure A

	segm_path: str,
		Path to tif file containing single cell image to be segmented for
		subfigure A

	bound_thres: int,
		Width of the boundary in pixels for subfigure A

	hole_size: int
		Size of the black hole in the center of the heat map

	nbins: int
		Number of bins to use for heatmap

	pixel_size : float
		Pixel size in um

	frame_rate : float
		Frame rate in frames per second (fps)

	divide_num : float
		Divide number to use for the MSD curves

	"""

	fig = plt.figure(figsize=(10,10))

	# """
	# ~~~~~~~~~~~Initialize Grid~~~~~~~~~~~~~~
	# """

	gs2 = GridSpec(6, 6)
	gs2.update(left=0.1, right=0.45, bottom=.55, top=.95, wspace=0, hspace=0)
	ax5 = plt.subplot(gs2[0:4, :3])
	ax6 = plt.subplot(gs2[0:2, 3:])
	ax7 = plt.subplot(gs2[2:4, 3:])

	gs1 = GridSpec(6, 6)
	gs1.update(left=0.45, right=0.925, bottom=.05, top=.95, wspace=.1, hspace=.05)
	ax1 = plt.subplot(gs1[0, :3], projection='polar')
	ax2 = plt.subplot(gs1[0, 3:], projection='polar')
	ax3 = plt.subplot(gs1[1, :3], projection='polar')
	ax4 = plt.subplot(gs1[1, 3:], projection='polar')

	gs3 = GridSpec(6, 6)
	gs3.update(left=0.1, right=0.48, bottom=.05, top=.6, wspace=40, hspace=30)
	ax8 = plt.subplot(gs3[0:4, :4])
	ax9 = plt.subplot(gs3[0:2, 4:])
	ax10 = plt.subplot(gs3[2:4, 4:])

	gs4 = GridSpec(6, 6)
	gs4.update(left=0.6, right=0.98, bottom=.05, top=.6, wspace=100, hspace=30)
	ax11 = plt.subplot(gs4[0:4, :4])
	ax12 = plt.subplot(gs4[0:2, 4:])
	ax13 = plt.subplot(gs4[2:4, 4:])
	df_cpy = deepcopy(df)

	# """
	# ~~~~~~~~~~~CTRL Data~~~~~~~~~~~~~~
	# """

	ctrl_df = df.loc[df['exp_label'] == 'Ctr']

	ctrl_df_bincenters, ctrl_df_binned = bin_df(ctrl_df,
												'avg_dist_bound',
												nbins=nbins)

	ctrl_df_binned = ctrl_df_binned.groupby(['category'])
	D_ctrl = ctrl_df_binned['D'].mean().to_numpy()
	alpha_ctrl = ctrl_df_binned['alpha'].mean().to_numpy()

	r_cont_ctrl, D_ctrl = interpolate_lin(ctrl_df_bincenters,
										  D_ctrl,
										  pad_size=hole_size)

	r_cont_ctrl, alpha_ctrl = interpolate_lin(ctrl_df_bincenters,
											  alpha_ctrl,
											  pad_size=hole_size)

	# """
	# ~~~~~~~~~~~BLM Data~~~~~~~~~~~~~~
	# """

	blm_df = df.loc[df['exp_label'] == 'BLM']

	blm_df_bincenters, blm_df_binned = bin_df(blm_df,
											  'avg_dist_bound',
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
	# ~~~~~~~~~~~Heat Maps~~~~~~~~~~~~~~
	# """

	D_all =  np.concatenate((D_ctrl, D_blm))
	min = D_all[D_all != 0].min()
	max = D_all[D_all != 0].max()
	xlabel = r'$\mathbf{D (nm^{2}/s)}$'
	ylabel = r'$\mathbf{D (nm^{2}/s)}$'

	add_heat_map(ax1,
				 ctrl_df_bincenters,
				 r_cont_ctrl,
				 D_ctrl,
				 ylabel=ylabel,
				 nbins=nbins,
				 hole_size=hole_size,
				 range=(min,max),
				 show_colorbar=True)

	add_heat_map(ax2,
				 blm_df_bincenters,
				 r_cont_blm,
				 D_blm,
				 ylabel=ylabel,
				 nbins=nbins,
				 hole_size=hole_size,
				 range=(min,max))

	xlabel = r'$\mathbf{\alpha}$'
	ylabel = r'$\mathbf{\alpha}$'

	alpha_all =  np.concatenate((alpha_ctrl, alpha_blm))
	min = alpha_all[alpha_all != 0].min()
	max = alpha_all[alpha_all != 0].max()

	add_heat_map(ax3,
				 ctrl_df_bincenters,
				 r_cont_ctrl,
				 alpha_ctrl,
				 ylabel=ylabel,
				 nbins=nbins,
				 hole_size=hole_size,
				 range=(min,max),
				 show_colorbar=True)

	add_heat_map(ax4,
				 blm_df_bincenters,
				 r_cont_blm,
				 alpha_blm,
				 ylabel=ylabel,
				 nbins=nbins,
				 hole_size=hole_size,
				 range=(min,max))

	ax1.set_title(r'$\mathbf{CTRL}$')
	ax2.set_title(r'$\mathbf{BLM}$')

	# """
	# ~~~~~~~~~~~CTRL MSD Curve~~~~~~~~~~~~~~
	# """

	ctrl_df_cpy = df_cpy.loc[df_cpy['exp_label'] == 'Ctr']

	add_mean_msd(ax8,
				 ctrl_df_cpy,
				 pixel_size,
				 frame_rate,
				 divide_num,
				 'sort_flag_boundary')

	ax8.set_title(r'$\mathbf{CTRL}$')

	# """
	# ~~~~~~~~~~~CTRL Strip Plots~~~~~~~~~~~~~~
	# """

	palette = color_palette("coolwarm", 7)
	palette = [palette[0], palette[-1]]
	add_strip_plot(ax9,
				   ctrl_df,
				   'D',
				   'sort_flag_boundary',
				   xlabels=['Interior', 'Boundary'],
				   ylabel=r'\mathbf{D (nm^{2}/s)}',
				   palette=palette,
				   x_labelsize=8,
				   drop_duplicates=True)

	add_t_test(ax9,
			   ctrl_df,
			   cat_col='sort_flag_boundary',
			   hist_col='D',
			   text_pos=[0.9, 0.9])

	ax = add_strip_plot(ax10,
						ctrl_df,
						'alpha',
						'sort_flag_boundary',
						xlabels=['Interior', 'Boundary'],
						ylabel=r'\mathbf{\alpha}',
						palette=palette,
						x_labelsize=8,
						drop_duplicates=True)

	add_t_test(ax10,
			   ctrl_df,
			   cat_col='sort_flag_boundary',
			   hist_col='alpha',
			   text_pos=[0.9, 0.9])


	# """
	# ~~~~~~~~~~~BLM MSD Curve~~~~~~~~~~~~~~
	# """

	blm_df_cpy = df_cpy.loc[df_cpy['exp_label'] == 'BLM']
	add_mean_msd(ax11,
				 blm_df_cpy,
				 pixel_size,
				 frame_rate,
				 divide_num,
				 'sort_flag_boundary',)

	ax11.set_title(r'$\mathbf{BLM}$')

	# """
	# ~~~~~~~~~~~BLM Strip Plot~~~~~~~~~~~~~~
	# """

	add_strip_plot(ax12,
				   blm_df,
				   'D',
				   'sort_flag_boundary',
				   xlabels=['Interior', 'Boundary'],
				   ylabel=r'\mathbf{D (nm^{2}/s)}',
				   palette=palette,
				   x_labelsize=8,
				   drop_duplicates=True)

	add_t_test(ax12,
			   blm_df,
			   cat_col='sort_flag_boundary',
			   hist_col='D',
			   text_pos=[0.9, 0.9])

	add_strip_plot(ax13,
				   blm_df,
				   'alpha',
				   'sort_flag_boundary',
				   xlabels=['Interior', 'Boundary'],
				   ylabel=r'\mathbf{\alpha}',
				   palette=palette,
				   x_labelsize=8,
				   drop_duplicates=True)

	add_t_test(ax13,
			   blm_df,
			   cat_col='sort_flag_boundary',
			   hist_col='alpha',
			   text_pos=[0.9, 0.9])

	# """
	# ~~~~~~~~~~~Data Selection Figure~~~~~~~~~~~~~
	# """

	df = pd.read_csv(df_path)
	im = imread(im_path)[0]
	segm = imread(segm_path)[0]
	mask = get_thres_mask(segm, sig=3, thres_rel=.05)
	mask = get_dist2boundary_mask(mask)

	im = gray2rgb(im, alpha=True)
	im_cpy = deepcopy(im)
	im[(mask <= 0) & (mask <= bound_thres)] = [0, 0, 100, 255]
	im[(mask < 0) & (mask > bound_thres)] = [100, 0, 0, 255]

	out = np.ubyte(0.7*im + 0.3*im_cpy)

	ax5.imshow(out)

	anno_traj(ax5,
			  df,
			  pixel_size=pixel_size,
			  show_traj_num=False,
			  cb_pos='left',
			  cb_tick_loc='left')
	ax5.set_aspect('auto')

	anno_traj(ax6,
			  df,
			  image=im,
			  pixel_size=pixel_size,
			  choose_particle=59,
			  show_traj_num=False,
			  show_colorbar=False)
	ax6.set_aspect('auto')

	anno_traj(ax7,
			  df,
			  image=im,
			  pixel_size=pixel_size,
			  choose_particle=118,
			  show_traj_num=False,
			  show_colorbar=False)
	ax7.set_aspect('auto')

	plt.tight_layout()
	plt.show()
