import pims; import pandas as pd; import numpy as np
import trackpy as tp
import os.path as osp; import os; import ast
from datetime import date, datetime
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_uint, img_as_int
from skimage.measure import regionprops
import warnings
import glob
import sys

from ..deno import filter_batch
from ..io import *
from ..segm import *
from ..video import *
from ..plot.plotutil import *
from ..regi import get_regi_params, apply_regi_params

from ..smt.detect import detect_blobs, detect_blobs_batch
from ..smt.fit_psf import fit_psf, fit_psf_batch
from ..smt.track import track_blobs
from ..smt.msd import plot_msd_batch, get_sorter_list
from ..smt import *
from ..phys import *
from ..plot import *
from ..plot import plot_phys_1 as plot_merged
from ..phys.physutil import relabel_particles, merge_physdfs


class Config():

	def __init__(self, config):

		#I/O
		self.INPUT_PATH = config['IO input_path']
		self.OUTPUT_PATH = config['IO output_path']
		self.ROOT_NAME = ''
		self.PIXEL_SIZE = config['Pixel_size']

		#CELL DENOISE SETTINGS
		self.CELL_MINIMUM_RADIUS = config['Cell deno minimum_radius']

		#CELL DETECTION SETTINGS
		self.CELL_THRESHOLD = config['Cell det blob_thres_rel']
		self.CELL_MIN_SIGMA = config['Cell det blob_min_sigma']
		self.CELL_MAX_SIGMA = config['Cell det blob_max_sigma']
		self.CELL_NUM_SIGMA = config['Cell det blob_num_sigma']
		self.CELL_PEAK_THRESH_REL = config['Cell det pk_thres_rel']
		self.CELL_R_TO_SIGRAW = config['Cell det r_to_sigraw']

		#CELL TRACKING SETTINGS
		self.CELL_SEARCH_RANGE = config['Cell trak search_range']
		self.CELL_MEMORY = config['Cell trak memory']
		self.CELL_TRAJ_LEN_THRES = config['Cell traj_length_thres']

		#DENOISE SETTINGS
		self.BOXCAR_RADIUS = config['Foci deno boxcar_radius']
		self.GAUS_BLUR_SIG = config['Foci deno gaus_blur_sig']

		#DETECTION SETTINGS
		self.THRESHOLD = config['Foci det blob_thres_rel']
		self.MIN_SIGMA = config['Foci det blob_min_sigma']
		self.MAX_SIGMA = config['Foci det blob_max_sigma']
		self.NUM_SIGMA = config['Foci det blob_num_sigma']
		self.PEAK_MIN = 0
		self.NUM_PEAKS = 1
		self.PEAK_THRESH_REL = config['Foci det pk_thres_rel']
		self.MASS_THRESH_REL = config['Foci det mass_thres_rel']
		self.PEAK_R_REL = 0
		self.MASS_R_REL = 0

		#TRACKING FILTERING SETTINGS
		if (config['Foci filt max_dist_err']=='') & \
			(config['Foci filt max_sig_to_sigraw']=='') & \
			(config['Foci filt max_delta_area']==''):
			self.DO_FILTER = False
		else:
			self.DO_FILTER = True

		self.FILTERS = {

		'MAX_DIST_ERROR': config['Foci filt max_dist_err'],
		'SIG_TO_SIGRAW' : config['Foci filt max_sig_to_sigraw'],
		'MAX_DELTA_AREA': config['Foci filt max_delta_area'],

		}

		#DICT
		self.DICT = config.copy()


	def save_config(self):

		path = self.OUTPUT_PATH + self.ROOT_NAME + '-analMeta.csv'
		config_df = pd.DataFrame.from_dict(data=self.DICT, orient='index')
		config_df = config_df.drop(['IO input_path', 'IO output_path',
									'Processed By:'])
		config_df.to_csv(path, header=False)

	def clean_dir(self):

		flist = [f for f in os.listdir(self.OUTPUT_PATH)]
		for f in flist:
		    os.remove(os.path.join(self.OUTPUT_PATH, f))


def file1_exists_or_pimsopen_file2(head_str, tail_str1, tail_str2):
	if osp.exists(head_str + tail_str1):
		frames = pims.open(head_str + tail_str1)
	else:
		frames = pims.open(head_str + tail_str2)
	return frames


class Pipeline3():

	def __init__(self, config):
		self.config = config

	def clean_dir(self):
		self.config.clean_dir()

	def load(self):
		if osp.exists(self.config.INPUT_PATH + self.config.ROOT_NAME + '.tif'):
			frames = imread(self.config.INPUT_PATH + self.config.ROOT_NAME + '.tif')
		else:
			frames = imread(self.config.INPUT_PATH + self.config.ROOT_NAME + '-raw.tif')

		# sides_pixel_num = 75
		# if sides_pixel_num:
		# 	new_shape = (frames.shape[0],
		# 				frames.shape[1] + sides_pixel_num*2,
		# 				frames.shape[2] + sides_pixel_num*2)
		# 	new_frames = np.zeros(new_shape, dtype=frames.dtype)
		# 	new_frames[:,
		# 		sides_pixel_num:sides_pixel_num+frames.shape[1],
		# 		sides_pixel_num:sides_pixel_num+frames.shape[2]] = frames
		# 	frames = new_frames

		frames = img_as_ubyte(frames)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif', frames)

	def cell_denoise(self):

		print("######################################")
		print('Applying Minimum Filter')
		print("######################################")

		frames = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif')
		filtered = filter_batch(frames, method='minimum', arg=self.config.CELL_MINIMUM_RADIUS)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-celldeno.tif', filtered)


	def check_cell_detection(self):
		print("######################################")
		print("Check cell detection")
		print("######################################")

		check_frame_ind = [0, 50, 100]

		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-celldeno.tif')

		for ind in check_frame_ind:
			blobs_df, det_plt_array = detect_blobs(frames[ind],
							min_sig=self.config.CELL_MIN_SIGMA,
							max_sig=self.config.CELL_MAX_SIGMA,
							num_sig=self.config.CELL_NUM_SIGMA,
							blob_thres_rel=self.config.CELL_THRESHOLD,
							peak_thres_rel=self.config.CELL_PEAK_THRESH_REL,
							overlap=0.5,
							r_to_sigraw=self.config.CELL_R_TO_SIGRAW,
							pixel_size=self.config.PIXEL_SIZE,
							diagnostic=True,
							pltshow=True,
							blob_markersize=10,
							plot_r=True,
							truth_df=None,
							)


	def detect_cell(self):
		print("######################################")
		print("Detect cell")
		print("######################################")

		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-raw.tif')
		frames_deno = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-celldeno.tif')

		blobs_df, det_plt_array = detect_blobs_batch(frames_deno,
									min_sig=self.config.CELL_MIN_SIGMA,
									max_sig=self.config.CELL_MAX_SIGMA,
									num_sig=self.config.CELL_NUM_SIGMA,
									blob_thres_rel=self.config.CELL_THRESHOLD,
									peak_thres_rel=self.config.CELL_PEAK_THRESH_REL,
									overlap=0.5,
									r_to_sigraw=self.config.CELL_R_TO_SIGRAW,
									pixel_size=self.config.PIXEL_SIZE,
									diagnostic=False,
									pltshow=False,
									plot_r=True,
									truth_df=None)

		det_plt_array = anim_blob(blobs_df, frames,
							show_image=True,
							pixel_size=self.config.PIXEL_SIZE,
							plot_r=True,
							figsize=(9, 9),
							dpi=100,
							)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-cellDetVideo.tif', det_plt_array)

		blobs_df = blobs_df.apply(pd.to_numeric)
		blobs_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-cellDetData.csv', index=False)

	def track_cell(self):
		det_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-cellDetData.csv')

		blobs_df = tp.link_df(det_df,
						    search_range=self.config.CELL_SEARCH_RANGE,
							memory=self.config.CELL_MEMORY,
							)
		blobs_df = tp.filter_stubs(blobs_df, 5)
		blobs_df = blobs_df.reset_index(drop=True)
		blobs_df = add_traj_length(blobs_df)

		traj_num_before = blobs_df['particle'].nunique()
		after_filter_df = blobs_df [blobs_df['traj_length'] >= self.config.CELL_TRAJ_LEN_THRES]
		print("######################################")
		print("Trajectory number before filters: \t%d" % traj_num_before)
		print("Trajectory number after filters: \t%d" % after_filter_df['particle'].nunique())
		print("######################################")

		blobs_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
									'-cellPhysData.csv', index=False)

		self.config.save_config()


	def plot_cell_traj(self):
		print("######################################")
		print("Plotting cell trajectories")
		print("######################################")

		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-raw.tif')
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
							'-cellPhysData.csv')

		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.config.CELL_TRAJ_LEN_THRES!=None:
			phys_df = phys_df[ phys_df['traj_length']>=self.config.CELL_TRAJ_LEN_THRES ]

		# """
		# ~~~~~~~~Optimize the colorbar format~~~~~~~~
		# """
		phys_df = relabel_particles(phys_df, col1='particle', col2='particle')
		if len(phys_df.drop_duplicates('particle')) > 3:
			particle_max = phys_df['particle'].max()
			particle_min = phys_df['particle'].min()
			particle_range = particle_max - particle_min
			cb_min=particle_min
			cb_max=particle_max
			cb_major_ticker=round(0.2*particle_range)
			cb_minor_ticker=round(0.2*particle_range)
		else:
			cb_min, cb_max, cb_major_ticker, cb_minor_ticker = None, None, None, None

		fig, ax = plt.subplots()
		anno_traj(ax, phys_df,

					show_image=True,
					image = frames[0],

					show_scalebar=True,
					pixel_size=self.config.PIXEL_SIZE,

					show_colorbar=True,
					cb_min=cb_min,
					cb_max=cb_max,
	                cb_major_ticker=cb_major_ticker,
					cb_minor_ticker=cb_minor_ticker,

		            show_traj_num=True,

					show_particle_label=False,
					)

		fig.savefig(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-cellTraj.pdf')
		plt.clf(); plt.close()
		# plt.show()


	def anim_cell_traj(self):

		print("######################################")
		print("Animating cell trajectories")
		print("######################################")

		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-raw.tif')
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
							'-cellPhysData.csv')
		phys_df = phys_df.drop('r', axis=1)
		phys_df['r'] = phys_df['sig_raw'] * self.config.CELL_R_TO_SIGRAW
		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.config.CELL_TRAJ_LEN_THRES!=None:
			phys_df = phys_df[ phys_df['traj_length']>=self.config.CELL_TRAJ_LEN_THRES ]
		# """
		# ~~~~~~~~check if phys_df is empty~~~~~~~~
		# """
		if phys_df.empty:
			print('phys_df is empty. No traj to animate!')
			return

		# """
		# ~~~~~~~~Optimize the colorbar format~~~~~~~~
		# """
		phys_df = relabel_particles(phys_df, col1='particle', col2='particle')
		if len(phys_df.drop_duplicates('particle')) > 3:
			particle_max = phys_df['particle'].max()
			particle_min = phys_df['particle'].min()
			particle_range = particle_max - particle_min
			cb_min=particle_min
			cb_max=particle_max
			cb_major_ticker=round(0.2*particle_range)
			cb_minor_ticker=round(0.2*particle_range)
		else:
			cb_min, cb_max, cb_major_ticker, cb_minor_ticker = None, None, None, None

		# """
		# ~~~~~~~~Generate and save animation video~~~~~~~~
		# """
		anim_tif = anim_traj(phys_df, frames,

					show_image=True,

					show_scalebar=True,
					pixel_size=self.config.PIXEL_SIZE,

					show_colorbar=True,
					cb_min=cb_min,
					cb_max=cb_max,
	                cb_major_ticker=cb_major_ticker,
					cb_minor_ticker=cb_minor_ticker,

					plot_r=True,

					show_traj_num=False,

		            show_tail=True,
					tail_length=50,

					show_boundary=False,

					dpi=100,
					)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-cellAnimVideo.tif', anim_tif)


	def blob_segm(self):
		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-raw.tif')
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
							'-cellPhysData.csv')
		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.config.CELL_TRAJ_LEN_THRES!=None:
			phys_df = phys_df[ phys_df['traj_length']>=self.config.CELL_TRAJ_LEN_THRES ]
		phys_df = phys_df.drop('r', axis=1)
		phys_df['r'] = phys_df['sig_raw'] * 2.5
		blob_segm(frames, phys_df,
				output_path_prefix=self.config.OUTPUT_PATH + self.config.ROOT_NAME,
				)


	def foci_denoise(self):

		print("######################################")
		print('Applying Boxcar Filter')
		print("######################################")

		frames = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif')
		frames = img_as_ubyte(frames)
		filtered = filter_batch(frames, method='boxcar', arg=self.config.BOXCAR_RADIUS)
		filtered = filter_batch(filtered, method='gaussian', arg=self.config.GAUS_BLUR_SIG)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-fociDeno.tif', filtered)


	def check_foci_detection(self):

		print("######################################")
		print("Check foci detection")
		print("######################################")

		check_frame_ind = [0, 50, 100, 144]

		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-raw.tif')

		for ind in check_frame_ind:
			blobs_df, det_plt_array = detect_blobs(frames[ind],
										min_sig=self.config.MIN_SIGMA,
										max_sig=self.config.MAX_SIGMA,
										num_sig=self.config.NUM_SIGMA,
										blob_thres_rel=self.config.THRESHOLD,
										peak_min=self.config.PEAK_MIN,
										num_peaks=self.config.NUM_PEAKS,
										peak_thres_rel=self.config.PEAK_THRESH_REL,
										mass_thres_rel=self.config.MASS_THRESH_REL,
										peak_r_rel=self.config.PEAK_R_REL,
										mass_r_rel=self.config.MASS_R_REL,
										r_to_sigraw=1,
										pixel_size=self.config.PIXEL_SIZE,
										diagnostic=True,
										pltshow=True,
										blob_markersize=5,
										plot_r=False,
										truth_df=None)

	def detect_foci(self):

		print("######################################")
		print("Detect foci")
		print("######################################")

		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-raw.tif')

		blobs_df, det_plt_array = detect_blobs_batch(frames,
									min_sig=self.config.MIN_SIGMA,
									max_sig=self.config.MAX_SIGMA,
									num_sig=self.config.NUM_SIGMA,
									blob_thres_rel=self.config.THRESHOLD,
									peak_min=self.config.PEAK_MIN,
									num_peaks=self.config.NUM_PEAKS,
									peak_thres_rel=self.config.PEAK_THRESH_REL,
									mass_thres_rel=self.config.MASS_THRESH_REL,
									peak_r_rel=self.config.PEAK_R_REL,
									mass_r_rel=self.config.MASS_R_REL,
									r_to_sigraw=1,
									pixel_size=self.config.PIXEL_SIZE,
									diagnostic=False,
									pltshow=False,
									blob_markersize=5,
									plot_r=False,
									truth_df=None)

		blobs_df = blobs_df[ blobs_df['frame']!=246 ]

		det_plt_array = anim_blob(blobs_df, frames,
									pixel_size=self.config.PIXEL_SIZE,
									blob_markersize=5,
									)
		try:
			imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-detVideo.tif', det_plt_array)
		except:
			pass

		blobs_df = blobs_df.apply(pd.to_numeric)
		blobs_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-detData.csv', index=False)

		self.config.DICT['Load existing analMeta'] = True
		self.config.save_config()


	def plot_foci_dynamics(self):

		print("######################################")
		print("Plotting foci dynamics")
		print("######################################")

		blobs_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-detData.csv')

		blobs_df = blobs_df[ blobs_df['frame']!=246 ]

		foci_dynamics_fig = plot_foci_dynamics(blobs_df)
		foci_dynamics_fig.savefig(self.config.OUTPUT_PATH + \
						self.config.ROOT_NAME + '-foci-dynamics.pdf')


	def fit(self):

		print("######################################")
		print("Fit")
		print("######################################")

		blobs_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-detData.csv')

		blobs_df = blobs_df[ blobs_df['frame']!=246 ]

		frames_deno = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-fociDeno.tif')

		psf_df, fit_plt_array = fit_psf_batch(frames_deno,
		            blobs_df,
		            diagnostic=False,
		            pltshow=False,
		            diag_max_dist_err=self.config.FILTERS['MAX_DIST_ERROR'],
		            diag_max_sig_to_sigraw = self.config.FILTERS['SIG_TO_SIGRAW'],
		            truth_df=None,
		            segm_df=None)

		psf_df = psf_df.apply(pd.to_numeric)
		psf_df['slope'] = psf_df['A'] / (9 * np.pi * psf_df['sig_x'] * psf_df['sig_y'])
		psf_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-fittData.csv', index=False)


	def plot_foci_dynamics2(self):
		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-raw.tif')
		fitt_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-fittData.csv')

		fitt_df = fitt_df[fitt_df['dist_err'] < self.config.DICT['Foci filt max_dist_err']]
		fitt_df = fitt_df[fitt_df['sigx_to_sigraw'] < self.config.DICT['Foci filt max_sig_to_sigraw']]
		fitt_df = fitt_df[fitt_df['sigy_to_sigraw'] < self.config.DICT['Foci filt max_sig_to_sigraw']]

		fitt_df = fitt_df[ fitt_df['frame']!=246 ]

		foci_dynamics_fig = plot_foci_dynamics(fitt_df)
		foci_dynamics_fig.savefig(self.config.OUTPUT_PATH + \
						self.config.ROOT_NAME + '-foci-dynamics2.pdf')

		fitt_plt_array = anim_blob(fitt_df, frames,
									pixel_size=self.config.PIXEL_SIZE,
									blob_markersize=5,
									)
		try:
			imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-fittVideo.tif', fitt_plt_array)
		except:
			pass


	def anim_traj(self):

		print("######################################")
		print("Animating trajectories")
		print("######################################")

		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = file1_exists_or_pimsopen_file2(self.config.OUTPUT_PATH + self.config.ROOT_NAME,
									'-regi.tif', '-raw.tif')
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.config.FILTERS['TRAJ_LEN_THRES']!=None:
			phys_df = phys_df[ phys_df['traj_length'] > self.config.FILTERS['TRAJ_LEN_THRES'] ]

		# """
		# ~~~~~~~~check if phys_df is empty~~~~~~~~
		# """
		if phys_df.empty:
			print('phys_df is empty. No traj to animate!')
			return

		phys_df = phys_df[ phys_df['frame']!=246 ]

		phys_plt_array = anim_blob(phys_df, frames,
									pixel_size=self.config.PIXEL_SIZE,
									blob_markersize=5,
									)
		try:
			imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physVideo.tif', phys_plt_array)
		except:
			pass

		foci_dynamics_fig = plot_foci_dynamics(phys_df)
		foci_dynamics_fig.savefig(self.config.OUTPUT_PATH + \
						self.config.ROOT_NAME + '-foci-dynamics3.pdf')


	# helper function for filt and track()
	def track_blobs_twice(self):
		frames = file1_exists_or_pimsopen_file2(self.config.OUTPUT_PATH + self.config.ROOT_NAME,
									'-regi.tif', '-raw.tif')
		psf_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-fittData.csv')


		blobs_df, im = track_blobs(psf_df,
								    search_range=self.config.SEARCH_RANGE,
									memory=self.config.MEMORY,
									pixel_size=self.config.PIXEL_SIZE,
									frame_rate=self.config.FRAME_RATE,
									divide_num=self.config.DIVIDE_NUM,
									filters=None,
									do_filter=False)

		if self.config.DO_FILTER:
			blobs_df, im = track_blobs(blobs_df,
									    search_range=self.config.SEARCH_RANGE,
										memory=self.config.MEMORY,
										pixel_size=self.config.PIXEL_SIZE,
										frame_rate=self.config.FRAME_RATE,
										divide_num=self.config.DIVIDE_NUM,
										filters=self.config.FILTERS,
										do_filter=True)

		# Add 'traj_length' column and save physData before traj_length_thres filter
		blobs_df = add_traj_length(blobs_df)

		return blobs_df


	# helper function for filt and track()
	def print_filt_traj_num(self, blobs_df):
		traj_num_before = blobs_df['particle'].nunique()
		after_filter_df = blobs_df [blobs_df['traj_length'] >= self.config.FILTERS['TRAJ_LEN_THRES']]
		print("######################################")
		print("Trajectory number before filters: \t%d" % traj_num_before)
		print("Trajectory number after filters: \t%d" % after_filter_df['particle'].nunique())
		print("######################################")


	# helper function for filt and track()
	def filt_phys_df(self, phys_df):

		df = phys_df.copy()
		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in df:
			df = df[ df['traj_length']>=self.config.FILTERS['TRAJ_LEN_THRES'] ]

		return df

	def filt_track(self):

		print("######################################")
		print("Filter and Linking")
		print("######################################")

		check_search_range = isinstance(self.config.SEARCH_RANGE, list)
		if check_search_range:
			param_list = self.config.SEARCH_RANGE
			particle_num_list = []
			phys_dfs = []
			mean_D_list = []
			mean_alpha_list = []
			for search_range in param_list:
				self.config.SEARCH_RANGE = search_range
				phys_df = self.track_blobs_twice()
				self.print_filt_traj_num(phys_df)
				phys_df = self.filt_phys_df(phys_df)
				phys_df = phys_df.drop_duplicates('particle')
				phys_df['search_range'] = search_range
				phys_dfs.append(phys_df)
				particle_num_list.append(len(phys_df))
				mean_D_list.append(phys_df['D'].mean())
				mean_alpha_list.append(phys_df['alpha'].mean())
			phys_df_all = pd.concat(phys_dfs)
			sr_opt_fig = plot_track_param_opt(
							track_param_name='search_range',
							track_param_unit='pixel',
							track_param_list=param_list,
							particle_num_list=particle_num_list,
							df=phys_df_all,
							mean_D_list=mean_D_list,
							mean_alpha_list=mean_alpha_list,
							)
			sr_opt_fig.savefig(self.config.OUTPUT_PATH + \
							self.config.ROOT_NAME + '-opt-search-range.pdf')


		check_traj_len_thres = isinstance(self.config.FILTERS['TRAJ_LEN_THRES'], list)
		if check_traj_len_thres:
			param_list = self.config.FILTERS['TRAJ_LEN_THRES']
			particle_num_list = []
			phys_dfs = []
			mean_D_list = []
			mean_alpha_list = []

			if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv'):
				original_phys_df = pd.read_csv(self.config.OUTPUT_PATH + \
					self.config.ROOT_NAME + '-physData.csv')
			else:
				original_phys_df = self.track_blobs_twice()

			for traj_len_thres in param_list:
				self.config.FILTERS['TRAJ_LEN_THRES'] = traj_len_thres
				self.print_filt_traj_num(original_phys_df)
				phys_df = self.filt_phys_df(original_phys_df)
				phys_df = phys_df.drop_duplicates('particle')
				phys_df['traj_len_thres'] = traj_len_thres
				phys_dfs.append(phys_df)
				particle_num_list.append(len(phys_df))
				mean_D_list.append(phys_df['D'].mean())
				mean_alpha_list.append(phys_df['alpha'].mean())
			phys_df_all = pd.concat(phys_dfs)
			sr_opt_fig = plot_track_param_opt(
							track_param_name='traj_len_thres',
							track_param_unit='frame',
							track_param_list=param_list,
							particle_num_list=particle_num_list,
							df=phys_df_all,
							mean_D_list=mean_D_list,
							mean_alpha_list=mean_alpha_list,
							)
			sr_opt_fig.savefig(self.config.OUTPUT_PATH + \
							self.config.ROOT_NAME + '-opt-traj-len-thres.pdf')

		else:
			blobs_df = self.track_blobs_twice()
			self.print_filt_traj_num(blobs_df)
			blobs_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
										'-physData.csv', index=False)



		self.config.DICT['Load existing analMeta'] = True
		self.config.save_config()

	def plot_traj(self):
		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = file1_exists_or_pimsopen_file2(self.config.OUTPUT_PATH + self.config.ROOT_NAME,
									'-regi.tif', '-raw.tif')
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.config.FILTERS['TRAJ_LEN_THRES']!=None:
			phys_df = phys_df[ phys_df['traj_length']>=self.config.FILTERS['TRAJ_LEN_THRES'] ]


		# """
		# ~~~~~~~~Optimize the colorbar format~~~~~~~~
		# """
		if len(phys_df.drop_duplicates('particle')) > 1:
			D_max = phys_df['D'].quantile(0.9)
			D_min = phys_df['D'].quantile(0.1)
			D_range = D_max - D_min
			cb_min=D_min
			cb_max=D_max
			cb_major_ticker=round(0.2*D_range)
			cb_minor_ticker=round(0.2*D_range)
		else:
			cb_min, cb_max, cb_major_ticker, cb_minor_ticker = None, None, None, None


		fig, ax = plt.subplots()
		anno_traj(ax, phys_df,

					show_image=True,
					image = frames[0],

					show_scalebar=True,
					pixel_size=self.config.PIXEL_SIZE,

					show_colorbar=True,
					cb_min=cb_min,
					cb_max=cb_max,
	                cb_major_ticker=cb_major_ticker,
					cb_minor_ticker=cb_minor_ticker,

		            show_traj_num=True,

					show_particle_label=False,

					# show_boundary=True,
					# boundary_mask=boundary_masks[0],
					# boundary_list=self.config.DICT['Sort dist_to_boundary'],
					)
		# fig.savefig(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-results.pdf')
		# plt.clf(); plt.close()
		plt.show()

		self.config.DICT['Load existing analMeta'] = True
		self.config.save_config()


	def merge_plot(self):

		start_ind = self.config.ROOT_NAME.find('_')
		end_ind = self.config.ROOT_NAME.find('_', start_ind+1)
		today = str(date.today().strftime("%y%m%d"))

		print("######################################")
		print("Merge and PlotMSD")
		print("######################################")

		merged_files = np.array(sorted(glob(self.config.OUTPUT_PATH + '/*physDataMerged.csv')))
		print(merged_files)

		if len(merged_files) > 1:
			print("######################################")
			print("Found multiple physDataMerged file!!!")
			print("######################################")
			return

		if len(merged_files) == 1:
			phys_df = pd.read_csv(merged_files[0])

		else:
			phys_files = np.array(sorted(glob(self.config.OUTPUT_PATH + '/*fittData.csv')))
			print("######################################")
			print("Total number of physData to be merged: %d" % len(phys_files))
			print("######################################")
			print(phys_files)

			if len(phys_files) > 1:
				ind = 1
				tot = len(phys_files)
				phys2_files = []
				for file in phys_files:
					print("Updating fittData (%d/%d)" % (ind, tot))
					ind = ind + 1

					curr_df = pd.read_csv(file, index_col=False)

					curr_df = curr_df[curr_df['dist_err'] < self.config.DICT['Foci filt max_dist_err']]
					curr_df = curr_df[curr_df['sigx_to_sigraw'] < self.config.DICT['Foci filt max_sig_to_sigraw']]
					curr_df = curr_df[curr_df['sigy_to_sigraw'] < self.config.DICT['Foci filt max_sig_to_sigraw']]
					curr_df = curr_df[ curr_df['frame']!=246 ]
					curr_df = add_foci_num(curr_df)
					curr_df = curr_df.drop_duplicates('frame')
					phys2_file = file[0:-4] + '2.csv'
					curr_df.round(3).to_csv(phys2_file, index=False)
					phys2_files.append(phys2_file)

				phys_df = merge_physdfs(phys2_files, mode='stiffness')
			else:
				phys_df = pd.read_csv(phys_files[0])


			phys_df.round(3).to_csv(self.config.OUTPUT_PATH + today + \
							'-physDataMerged.csv', index=False)

		stiffness_fig = plot_stiffness_kinetics(phys_df)
		stiffness_fig.savefig(self.config.OUTPUT_PATH + \
						today + '_stiffness-results.pdf')

		sys.exit()

def get_root_name_list(settings_dict):
	settings = settings_dict.copy()

	root_name_list = []
	path_list = glob(settings['IO input_path'] + '/*-fittData.csv')
	merge_list = glob(settings['IO input_path'] + '/*-physDataMerged.csv')
	if len(path_list) != 0:
		for path in path_list:
			temp = path.split('/')[-1]
			temp = temp[:-len('-fittData.csv')]
			root_name_list.append(temp)

	elif len(merge_list) == 1:
		for mergeData in merge_list:
			root_name_list.append(mergeData)

	else:
		path_list = glob(settings['IO input_path'] + '/*-raw.tif')
		if len(path_list) != 0:
			for path in path_list:
				temp = path.split('/')[-1]
				temp = temp[:-4 - len('-raw')]
				root_name_list.append(temp)
		else:
			path_list = glob(settings['IO input_path'] + '/*.tif')
			for path in path_list:
				temp = path.split('/')[-1]
				temp = temp[:-4]
				root_name_list.append(temp)

	return np.array(sorted(root_name_list))


def pipeline_batch(settings_dict, control_list):

	# """
	# ~~~~~~~~~~~~~~~~~1. Get root_name_list~~~~~~~~~~~~~~~~~
	# """
	root_name_list = get_root_name_list(settings_dict)

	print("######################################")
	print("Total data num to be processed: %d" % len(root_name_list))
	print(root_name_list)
	print("######################################")

	ind = 0
	tot = len(root_name_list)
	for root_name in root_name_list:
		ind = ind + 1
		print("\n")
		print("Processing (%d/%d): %s" % (ind, tot, root_name))

		# """
		# ~~~~~~~~~~~~~~~~~2. Update config~~~~~~~~~~~~~~~~~
		# """

		config = Config(settings_dict)
		config.ROOT_NAME = root_name
		config.DICT['Raw data file'] = root_name + '.tif'
		config.DICT['Processed date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		config.DICT['Processed by:'] = settings_dict['Processed By:']


		# """
		# ~~~~~~~~~~~~~~~~~3. Setup pipe and run~~~~~~~~~~~~~~~~~
		# """
		pipe = Pipeline3(config)
		for func in control_list:
			getattr(pipe, func)()
