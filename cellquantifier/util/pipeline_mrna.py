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
from ..phys import *
from ..plot import *
from ..plot import plot_phys_1 as plot_merged
from ..phys.physutil import relabel_particles, merge_physdfs

class Config():

	def __init__(self, config):

		#GENERAL INFO
		self.INPUT_PATH = config['IO input_path']
		self.OUTPUT_PATH = config['IO output_path']
		self.ROOT_NAME = ''
		self.PIXEL_SIZE = config['Pixel_size']
		self.FRAME_RATE = config['Frame_rate']

		#Mask SETTINGS
		self.DIST2BOUNDARY_MASK_NAME = ''
		self.MASK_SIG_BOUNDARY = config['Mask boundary_mask sig']
		self.MASK_THRES_BOUNDARY = config['Mask boundary_mask thres_rel']

		#DENOISE SETTINGS
		self.BOXCAR_RADIUS = config['Deno boxcar_radius']
		self.GAUS_BLUR_SIG = config['Deno gaus_blur_sig']

		#DETECTION SETTINGS
		self.THRESHOLD = config['Det blob_threshold']
		self.MIN_SIGMA = config['Det blob_min_sigma']
		self.MAX_SIGMA = config['Det blob_max_sigma']
		self.NUM_SIGMA = config['Det blob_num_sigma']
		self.PEAK_THRESH_REL = config['Det pk_thresh_rel']

		#mRNA TRACKING SETTINGS
		self.mRNA_SEARCH_RANGE = config['mRNA trak search_range']
		self.mRNA_MEMORY = config['mRNA trak memory']
		self.mRNA_DIVIDE_NUM = config['mRNA trak divide_num']
		self.mRNA_TRAJ_LEN_THRES = config['mRNA traj_length_thres']

	    #TRACKING SETTINGS
		self.SEARCH_RANGE = config['Trak search_range']
		self.MEMORY = config['Trak memory']
		self.DIVIDE_NUM = config['Trak divide_num']

		#TRACKING FILTERING SETTINGS
		if (config['Filt max_dist_err']=='') & \
			(config['Filt max_sig_to_sigraw']=='') & \
			(config['Filt max_delta_area']=='') & \
			(config['Filt traj_length_thres']==''):
			self.DO_FILTER = False
		else:
			self.DO_FILTER = True

		self.FILTERS = {

		'MAX_DIST_ERROR': config['Filt max_dist_err'],
		'SIG_TO_SIGRAW' : config['Filt max_sig_to_sigraw'],
		'MAX_DELTA_AREA': config['Filt max_delta_area'],
		'TRAJ_LEN_THRES': config['Filt traj_length_thres']

		}

		#SORTING SETTINGS
		if (config['Sort dist_to_boundary']==''):
			self.DO_SORT = False
		else:
			self.DO_SORT = True

		self.SORTERS = {

		'DIST_TO_BOUNDARY': config['Sort dist_to_boundary'],
		'TRAVEL_DIST' : config['Sort travel_dist']

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


def nonempty_exists_then_copy(input_path, output_path, filename):
	not_empty = len(filename)!=0
	exists_in_input = osp.exists(input_path + filename)

	if not_empty and exists_in_input:
		frames = imread(input_path + filename)
		frames = img_as_ubyte(frames)
		imsave(output_path + filename, frames)

def nonempty_openfile1_or_openfile2(path, filename1, filename2):
	if filename1 and osp.exists(path + filename1): # if not empty and exists
		frames = imread(path + filename1)
	else:
		frames = imread(path + filename2)
	return frames


class Pipeline3():

	def __init__(self, config):
		self.config = config

	def clean_dir(self):
		self.config.clean_dir()

	def load(self):
		# load data file
		if osp.exists(self.config.INPUT_PATH + self.config.ROOT_NAME + '.tif'):
			frames = imread(self.config.INPUT_PATH + self.config.ROOT_NAME + '.tif')
		else:
			frames = imread(self.config.INPUT_PATH + self.config.ROOT_NAME + '-raw.tif')

		frames = img_as_ubyte(frames)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif', frames)

		print('\tMask_dist2boundary_file: [%s]' % self.config.DIST2BOUNDARY_MASK_NAME)
		nonempty_exists_then_copy(self.config.INPUT_PATH, self.config.OUTPUT_PATH, self.config.DIST2BOUNDARY_MASK_NAME)

	def get_boundary_mask(self):
		# If no mask ref file, use raw file automatically
		frames = nonempty_openfile1_or_openfile2(self.config.OUTPUT_PATH,
					self.config.DIST2BOUNDARY_MASK_NAME,
					self.config.ROOT_NAME+'-raw.tif')

		# If only 1 frame available, duplicate it to enough frames_num.
		tot_frame_num = len(imread(self.config.OUTPUT_PATH + \
					self.config.ROOT_NAME+'-raw.tif'))
		if frames.ndim==2:
			dup_frames = np.zeros((tot_frame_num, frames.shape[0], frames.shape[1]),
									dtype=frames.dtype)
			for i in range(tot_frame_num):
				dup_frames[i] = frames
			frames = dup_frames

		boundary_masks = get_thres_mask_batch(frames,
					self.config.MASK_SIG_BOUNDARY, self.config.MASK_THRES_BOUNDARY)

		return boundary_masks

	def mask_boundary(self):
		print("######################################")
		print("Generate mask_boundary")
		print("######################################")
		boundary_masks = self.get_boundary_mask()
		# Save it using 255 and 0
		boundary_masks_255 = np.rint(boundary_masks / \
							boundary_masks.max() * 255).astype(np.uint8)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-boundaryMask.tif',
				boundary_masks_255)

		print("######################################")
		print("Generate dist2boundary_mask")
		print("######################################")
		dist2boundary_masks = img_as_int(get_dist2boundary_mask_batch(boundary_masks))
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist2boundaryMask.tif',
				dist2boundary_masks)

	def deno_gaus(self):

		print("######################################")
		print('Applying Boxcar and Gaussian Filter')
		print("######################################")

		frames = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif')
		filtered = filter_batch(frames, method='boxcar', arg=self.config.BOXCAR_RADIUS)
		filtered = filter_batch(filtered, method='gaussian', arg=self.config.GAUS_BLUR_SIG)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-deno.tif', filtered)


	def check_detect_fit(self):

		print("######################################")
		print("Check detection and fitting")
		print("######################################")

		check_frame_ind = [0, 100]

		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
									'-raw.tif')

		for ind in check_frame_ind:
			blobs_df, det_plt_array = detect_blobs(frames[ind],
										min_sig=self.config.MIN_SIGMA,
										max_sig=self.config.MAX_SIGMA,
										num_sig=self.config.NUM_SIGMA,
										blob_thres_rel=self.config.THRESHOLD,
										overlap=0.5,

										peak_thres_rel=self.config.PEAK_THRESH_REL,
										r_to_sigraw=1,
										pixel_size=self.config.PIXEL_SIZE,

										diagnostic=True,
										pltshow=True,
										plot_r=True,
										truth_df=None)

	def detect(self):

		print("######################################")
		print("Detect")
		print("######################################")

		frames = pims.open(self.config.OUTPUT_PATH + \
				self.config.ROOT_NAME + '-raw.tif')

		blobs_df, det_plt_array = detect_blobs_batch(frames,
									min_sig=self.config.MIN_SIGMA,
									max_sig=self.config.MAX_SIGMA,
									num_sig=self.config.NUM_SIGMA,
									blob_thres_rel=self.config.THRESHOLD,
									overlap=0,

									peak_thres_rel=self.config.PEAK_THRESH_REL,
									r_to_sigraw=1,
									pixel_size=self.config.PIXEL_SIZE,

									diagnostic=False,
									pltshow=False,
									plot_r=True,
									truth_df=None)

		blobs_df = blobs_df.apply(pd.to_numeric)
		blobs_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-detData.csv', index=False)


		det_plt_array = anim_blob(blobs_df, frames,
									pixel_size=self.config.PIXEL_SIZE,
									blob_markersize=5,
									)
		try:
			imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-detVideo.tif', det_plt_array)
		except:
			pass

		self.config.save_config()


	def track_mrna(self):
		det_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-detData.csv')

		blobs_df = tp.link_df(det_df,
						    search_range=self.config.mRNA_SEARCH_RANGE,
							memory=self.config.mRNA_MEMORY,
							)
		blobs_df = tp.filter_stubs(blobs_df, 5)
		blobs_df = blobs_df.reset_index(drop=True)
		blobs_df = add_traj_length(blobs_df)

		blobs_df_cut = blobs_df[['frame', 'x', 'y', 'particle']]
		blobs_df_cut = blobs_df_cut.apply(pd.to_numeric)
		im = tp.imsd(blobs_df_cut,
					mpp=self.config.PIXEL_SIZE,
					fps=self.config.FRAME_RATE,
					max_lagtime=np.inf,
					)

		blobs_df = get_d_values(blobs_df, im, self.config.mRNA_DIVIDE_NUM)
		blobs_df = blobs_df.apply(pd.to_numeric)

		traj_num_before = blobs_df['particle'].nunique()
		after_filter_df = blobs_df [blobs_df['traj_length'] >= self.config.mRNA_TRAJ_LEN_THRES]
		print("######################################")
		print("Trajectory number before filters: \t%d" % traj_num_before)
		print("Trajectory number after filters: \t%d" % after_filter_df['particle'].nunique())
		print("######################################")

		blobs_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
									'-physData.csv', index=False)

		self.config.save_config()


	def plot_mrna_traj(self):
		print("######################################")
		print("Plotting cell trajectories")
		print("######################################")

		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-base.tif')
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
							'-physData.csv')

		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.config.mRNA_TRAJ_LEN_THRES!=None:
			phys_df = phys_df[ phys_df['traj_length']>=self.config.mRNA_TRAJ_LEN_THRES ]

		# """
		# ~~~~~~~~Optimize the colorbar format~~~~~~~~
		# """
		# phys_df = relabel_particles(phys_df, col1='particle', col2='particle')
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

					show_scalebar=False,
					pixel_size=self.config.PIXEL_SIZE,

					show_colorbar=False,
					cb_min=cb_min,
					cb_max=cb_max,
	                cb_major_ticker=cb_major_ticker,
					cb_minor_ticker=cb_minor_ticker,

		            show_traj_num=False,

					show_particle_label=False,
					)

		fig.savefig(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-traj.tiff', dpi=600)
		# plt.clf(); plt.close()
		plt.show()


	def anim_mrna_traj(self):

		print("######################################")
		print("Animating cell trajectories")
		print("######################################")

		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
		 					'-deno.tif')
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
							'-physData.csv')

		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.config.mRNA_TRAJ_LEN_THRES!=None:
			phys_df = phys_df[ phys_df['traj_length']>=self.config.mRNA_TRAJ_LEN_THRES ]
		# """
		# ~~~~~~~~check if phys_df is empty~~~~~~~~
		# """
		if phys_df.empty:
			print('phys_df is empty. No traj to animate!')
			return

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

		# """
		# ~~~~~~~~Generate and save animation video~~~~~~~~
		# """
		anim_tif = anim_traj(phys_df, frames,

					show_image=True,

					show_scalebar=False,
					pixel_size=self.config.PIXEL_SIZE,

					show_colorbar=False,
					cb_min=cb_min,
					cb_max=cb_max,
	                cb_major_ticker=cb_major_ticker,
					cb_minor_ticker=cb_minor_ticker,

					plot_r=False,

					show_traj_num=False,

		            show_tail=True,
					tail_length=50,

					show_boundary=False,

					dpi=100,
					)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-animVideo.tif', anim_tif)


	def fit(self):

		print("######################################")
		print("Fit")
		print("######################################")

		blobs_df = pd.read_csv(self.config.OUTPUT_PATH + \
					self.config.ROOT_NAME + '-detData.csv')
		frames_deno = pims.open(self.config.OUTPUT_PATH + \
					self.config.ROOT_NAME + '-deno.tif')

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

		foci_prop_hist_fig = plot_foci_prop_hist(psf_df)
		foci_prop_hist_fig.savefig(self.config.OUTPUT_PATH + \
						self.config.ROOT_NAME + '-foci-prop-hist.pdf')


	# helper function for filt_track()
	def track_blobs_twice(self):
		psf_df = pd.read_csv(self.config.OUTPUT_PATH + \
				self.config.ROOT_NAME + '-fittData.csv')

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


	# helper function for filt_track()
	def print_filt_traj_num(self, blobs_df):
		traj_num_before = blobs_df['particle'].nunique()
		after_filter_df = blobs_df [blobs_df['traj_length'] >= self.config.FILTERS['TRAJ_LEN_THRES']]
		print("######################################")
		print("Trajectory number before filters: \t%d" % traj_num_before)
		print("Trajectory number after filters: \t%d" % after_filter_df['particle'].nunique())
		print("######################################")


	# helper function for filt_track()
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

		self.config.save_config()


	def plot_traj(self):
		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = pims.open(self.config.OUTPUT_PATH + \
					self.config.ROOT_NAME + '-raw.tif')
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + \
					self.config.ROOT_NAME + '-physData.csv')

		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.config.FILTERS['TRAJ_LEN_THRES']!=None:
			phys_df = phys_df[ phys_df['traj_length']>=self.config.FILTERS['TRAJ_LEN_THRES'] ]

		# # """
		# # ~~~~~~~~boudary_type filter~~~~~~~~
		# # """
		# if 'boundary_type' in phys_df:
		# 	phys_df = phys_df[ phys_df['boundary_type']!='--none--']

		# # """
		# # ~~~~~~~~travel_dist filter~~~~~~~~
		# # """
		# if 'travel_dist' in phys_df and self.config.DICT['Sort travel_dist']!=None:
		# 	travel_dist_min = self.config.DICT['Sort travel_dist'][0]
		# 	travel_dist_max = self.config.DICT['Sort travel_dist'][1]
		# 	phys_df = phys_df[ (phys_df['travel_dist']>=travel_dist_min) & \
		# 						(phys_df['travel_dist']<=travel_dist_max) ]

		# # """
		# # ~~~~~~~~particle_type filter~~~~~~~~
		# # """
		# if 'particle_type' in phys_df:
		# 	phys_df = phys_df[ phys_df['particle_type']!='--none--']


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


		# # """
		# # ~~~~~~~~Prepare the boundary_masks~~~~~~~~
		# # """
		# if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-boundaryMask.tif'):
		# 	boundary_masks = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-boundaryMask.tif')
		# 	boundary_masks = boundary_masks // 255
		# else:
		# 	boundary_masks = self.get_boundary_mask()


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
		fig.savefig(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-results.pdf', dpi=300)
		# plt.clf(); plt.close()
		plt.show()


	def anim_traj(self):

		print("######################################")
		print("Animating trajectories")
		print("######################################")

		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = pims.open(self.config.OUTPUT_PATH + \
				self.config.ROOT_NAME + '-raw.tif')
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + \
				self.config.ROOT_NAME + '-physData.csv')

		# """
		# ~~~~~~~~traj_length filter~~~~~~~~
		# """
		if 'traj_length' in phys_df and self.config.FILTERS['TRAJ_LEN_THRES']!=None:
			phys_df = phys_df[ phys_df['traj_length'] > self.config.FILTERS['TRAJ_LEN_THRES'] ]

		# # """
		# # ~~~~~~~~boudary_type filter~~~~~~~~
		# # """
		# if 'boundary_type' in phys_df:
		# 	phys_df = phys_df[ phys_df['boundary_type']!='--none--']

		# # """
		# # ~~~~~~~~travel_dist filter~~~~~~~~
		# # """
		# if 'travel_dist' in phys_df and self.config.DICT['Sort travel_dist']!=None:
		# 	travel_dist_min = self.config.DICT['Sort travel_dist'][0]
		# 	travel_dist_max = self.config.DICT['Sort travel_dist'][1]
		# 	phys_df = phys_df[ (phys_df['travel_dist']>travel_dist_min) & \
		# 						(phys_df['travel_dist']<travel_dist_max) ]

		# """
		# ~~~~~~~~particle_type filter~~~~~~~~
		# """
		if 'particle_type' in phys_df:
			phys_df = phys_df[ phys_df['particle_type']!='--none--']

		# """
		# ~~~~~~~~check if phys_df is empty~~~~~~~~
		# """
		if phys_df.empty:
			print('phys_df is empty. No traj to animate!')
			return

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

		# """
		# ~~~~~~~~Prepare the boundary_masks~~~~~~~~
		# """
		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-boundaryMask.tif'):
			boundary_masks = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-boundaryMask.tif')
			boundary_masks = boundary_masks // 255
		else:
			boundary_masks = self.get_boundary_mask()

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

					show_traj_num=True,

		            show_tail=True,
					tail_length=500,

					show_boundary=False,
					boundary_masks=boundary_masks,

					dpi=100,
					)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-animVideo.tif', anim_tif)


	def phys_dist2boundary(self):
		print("######################################")
		print("Add Physics Param: dist_to_boundary")
		print("######################################")
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist2boundaryMask.tif'):
			dist2boundary_masks = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-dist2boundaryMask.tif')
		else:
			boundary_masks = self.get_boundary_mask()
			dist2boundary_masks = get_dist2boundary_mask_batch(boundary_masks)

		phys_df = add_dist_to_boundary_batch_2(phys_df, dist2boundary_masks)
		phys_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)

	def phys_antigen_data(self):
		print("######################################")
		print("Add Physics Param: antigen_data")
		print("######################################")

		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		# phys_df['pixel_size'] = self.config.PIXEL_SIZE
		# phys_df['dist_to_boundary'] = phys_df['dist_to_boundary'] * self.config.PIXEL_SIZE
		# phys_df['dist_to_53bp1'] = phys_df['dist_to_53bp1'] * self.config.PIXEL_SIZE

		phys_df = add_antigen_data(phys_df, sorters=self.config.SORTERS)

		phys_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)

	def phys_antigen_data2(self):
		print("######################################")
		print("Add Physics Param: antigen_data2")
		print("######################################")

		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		phys_df = add_local_D_alpha(phys_df,
						pixel_size=self.config.PIXEL_SIZE,
						frame_rate=self.config.FRAME_RATE,
						window_width=20,
						divide_num=5,
						)

		phys_df = add_directional_persistence(phys_df,
						window_width=5,
						)

		phys_df.round(3).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)


	def plot_stub_hist(self):
		print("######################################")
		print("Plot")
		print("######################################")

		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		stub_hist_fig = plot_stub_prop_hist(phys_df)
		stub_hist_fig.savefig(self.config.OUTPUT_PATH + \
						self.config.ROOT_NAME + '-stub-prop-hist.pdf')


	def classify_antigen(self):
		print("######################################")
		print("Classify antigen sub_trajectories")
		print("######################################")

		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		phys_df = classify_antigen(phys_df)

		phys_df.round(3).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData2.csv', index=False)


	def plot_subtraj(self,
					subtype='DM',
					sp_traj_len_thres=None,
					sp_travel_dist_thres=None,
					):
		# """
		# ~~~~~~~~Prepare frames, phys_df~~~~~~~~
		# """
		frames = pims.open(self.config.OUTPUT_PATH + \
					self.config.ROOT_NAME + '-raw.tif')
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + \
					self.config.ROOT_NAME + '-physData2.csv')

		# # """
		# # ~~~~~~~~traj_length filter~~~~~~~~
		# # """
		# if 'traj_length' in phys_df and self.config.FILTERS['TRAJ_LEN_THRES']!=None:
		# 	phys_df = phys_df[ phys_df['traj_length']>=self.config.FILTERS['TRAJ_LEN_THRES'] ]


		phys_df = phys_df[ phys_df['subparticle_type']==subtype ]
		if sp_traj_len_thres:
			phys_df = phys_df[ phys_df['subparticle_traj_length']>=sp_traj_len_thres ]
		if sp_travel_dist_thres:
			phys_df = phys_df[ phys_df['subparticle_travel_dist']>=sp_travel_dist_thres ]

		phys_df = phys_df.rename(columns={
				'D':'original_D',
				'alpha':'original_alpha',
				'traj_length':'orig_traj_length',
				'particle':'original_particle',
				'subparticle': 'particle',
				'subparticle_D': 'D',
				})


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
		fig.savefig(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-' \
					+ subtype + '-trajs.pdf', dpi=300)
		# plt.clf(); plt.close()
		plt.show()

	def plot_DM_traj(self):
		self.plot_subtraj(subtype='DM',
					# sp_traj_len_thres=20,
					# sp_travel_dist_thres=10,
					)

	def plot_BM_traj(self):
		self.plot_subtraj(subtype='BM',
					sp_traj_len_thres=20,
					)

	def plot_CM_traj(self):
		self.plot_subtraj(subtype='CM',
					sp_traj_len_thres=20,
					)


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
			phys_files = np.array(sorted(glob(self.config.OUTPUT_PATH + '/*physData.csv')))
			print("######################################")
			print("Total number of physData to be merged: %d" % len(phys_files))
			print("######################################")
			print(phys_files)

			if len(phys_files) > 1:
				phys_df = merge_physdfs(phys_files, mode='general')
				phys_df = relabel_particles(phys_df)
			else:
				phys_df = pd.read_csv(phys_files[0])

			phys_df.round(6).to_csv(self.config.OUTPUT_PATH + merged_name + \
							'-physDataMerged.csv', index=False)

		# Apply traj_length_thres filter
		if 'traj_length' in phys_df:
			phys_df = phys_df[ phys_df['traj_length'] > self.config.FILTERS['TRAJ_LEN_THRES'] ]

		fig = plot_merged(phys_df, 'exp_label',
						pixel_size=self.config.PIXEL_SIZE,
						frame_rate=self.config.FRAME_RATE,
						divide_num=self.config.DIVIDE_NUM,
						RGBA_alpha=1,
						do_gmm=False)

		fig.savefig(self.config.OUTPUT_PATH + today + '-antigen-results.pdf')

		sys.exit()

def get_root_name_list(settings_dict):
	# Make a copy of settings_dict
	# Use '*%#@)9_@*#@_@' to substitute if the labels are empty
	settings = settings_dict.copy()

	if settings['Mask boundary_mask file label'] == '':
		settings['Mask boundary_mask file label'] = '*%#@)9_@*#@_@'

	root_name_list = []

	path_list = glob(settings['IO input_path'] + '/*-physData.csv')
	if len(path_list) != 0:
		for path in path_list:
			temp = path.split('/')[-1]
			temp = temp[:-len('-physData.csv')]
			root_name_list.append(temp)

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
				if (settings['Mask boundary_mask file label'] not in temp+'.tif'):
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

		# Update config.ROOT_NAME and config.DICT
		config.ROOT_NAME = root_name
		config.DICT['Raw data file'] = root_name + '.tif'
		config.DICT['Processed date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		config.DICT['Processed by:'] = settings_dict['Processed By:']

		# Update config.DIST2BOUNDARY_MASK_NAME
		if '-' in root_name and root_name.find('-')>0:
			key = root_name[0:root_name.find('-')]
		else:
			key = root_name

		if settings_dict['Mask boundary_mask file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' + key +
					'*' + settings_dict['Mask boundary_mask file label'] + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.DIST2BOUNDARY_MASK_NAME = file_list[0].split('/')[-1]
			else:
				config.DICT['Mask boundary_mask file label'] = ''

		# """
		# ~~~~~~~~~~~~~~~~~3. Setup pipe and run~~~~~~~~~~~~~~~~~
		# """
		pipe = Pipeline3(config)
		for func in control_list:
			getattr(pipe, func)()
