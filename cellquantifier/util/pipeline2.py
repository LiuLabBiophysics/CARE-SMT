import pims; import pandas as pd; import numpy as np
import trackpy as tp
import os.path as osp; import os; import ast
from datetime import date, datetime
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.measure import regionprops
import warnings
from glob import glob

from ..deno import filter_batch
from ..io import *
from ..segm import *
from ..regi import get_regi_params, apply_regi_params
from ..plot.plotutil import anno_raw

from ..smt.fit_psf import fit_psf, fit_psf_batch
from ..smt.track import track_blobs
from ..smt.msd import plot_msd_batch, get_sorter_list
from ..phys import *
from ..util.config2 import Config
from ..plot import plot_phys_1 as plot_merged
from ..phys.physutil import relabel_particles, merge_physdfs

from ..plot.plotutil import anno_blob, anno_scatter
from ..plot.plotutil import plot_end
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def detect_blobs(pims_frame,
				min_sig=1,
				max_sig=3,
				num_sig=5,
				blob_thres=0.1,
				peak_thres_rel=0.1,
				r_to_sigraw=3,
				pixel_size = .1084,
				diagnostic=True,
				pltshow=True,
				plot_r=True,
				blob_marker='^',
				blob_markersize=10,
				blob_markercolor=(0,0,1,0.8),
				truth_df=None):

	# """
	# ~~~~~~~~~~~~~~~~~Detection using skimage.feature.blob_log~~~~~~~~~~~~~~~~~
	# """

	frame = pims_frame
	blobs = blob_log(frame,
					 min_sigma=min_sig,
					 max_sigma=max_sig,
					 num_sigma=num_sig,
					 threshold=blob_thres)

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~Prepare blobs_df and update it~~~~~~~~~~~~~~~~~~~~~~
	# """

	columns = ['frame', 'x', 'y', 'sig_raw', 'r', 'peak']
	blobs_df = pd.DataFrame([], columns=columns)
	blobs_df['x'] = blobs[:, 0]
	blobs_df['y'] = blobs[:, 1]
	blobs_df['sig_raw'] = blobs[:, 2]
	blobs_df['r'] = blobs[:, 2] * r_to_sigraw
	blobs_df['frame'] = pims_frame.frame_no
	# """
	# ~~~~~~~Filter detections at the edge~~~~~~~
	# """
	blobs_df = blobs_df[(blobs_df['x'] - blobs_df['r'] > 0) &
				  (blobs_df['x'] + blobs_df['r'] + 1 < frame.shape[0]) &
				  (blobs_df['y'] - blobs_df['r'] > 0) &
				  (blobs_df['y'] + blobs_df['r'] + 1 < frame.shape[1])]
	for i in blobs_df.index:
		x = int(blobs_df.at[i, 'x'])
		y = int(blobs_df.at[i, 'y'])
		r = int(round(blobs_df.at[i, 'r']))
		blob = frame[x-r:x+r+1, y-r:y+r+1]
		blobs_df.at[i, 'peak'] = blob.max()

	# """
	# ~~~~~~~Filter detections below peak_thres_abs~~~~~~~
	# """

	peak_thres_abs = blobs_df['peak'].max() * peak_thres_rel
	blobs_df = blobs_df[(blobs_df['peak'] > peak_thres_abs)]

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~Print detection summary~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	if len(blobs_df)==0:
		print("\n"*3)
		print("##############################################")
		print("ERROR: No blobs detected in this frame!!!")
		print("##############################################")
		print("\n"*3)
		return pd.DataFrame(np.array([])), np.array([])
	else:
		print("Det in frame %d: %s" % (pims_frame.frame_no, len(blobs_df)))

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Diagnostic~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	plt_array = []
	if diagnostic:
		fig, ax = plt.subplots(figsize=(9,9))

		# """
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~Annotate the blobs~~~~~~~~~~~~~~~~~~~~~~~~~~
		# """
		ax.imshow(frame, cmap="gray", aspect='equal')
		anno_blob(ax, blobs_df, marker=blob_marker, markersize=blob_markersize,
				plot_r=plot_r, color=blob_markercolor)

		# """
		# ~~~~~~~~~~~~~~~~~~~Annotate ground truth if needed~~~~~~~~~~~~~~~~~~~
		# """
		if isinstance(truth_df, pd.DataFrame):
			anno_scatter(ax, truth_df, marker='o', color=(0,1,0,0.8))

		# """
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Add scale bar~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# """
		font = {'family': 'arial', 'weight': 'bold','size': 16}
		scalebar = ScaleBar(pixel_size, 'um', location = 'upper right',
			font_properties=font, box_color = 'black', color='white')
		scalebar.length_fraction = .3
		scalebar.height_fraction = .025
		ax.add_artist(scalebar)

		plt_array = plot_end(fig, pltshow)

	return blobs_df, plt_array


def detect_blobs_batch(pims_frames,
			min_sig=1,
			max_sig=3,
			num_sig=5,
			blob_thres=0.1,
			peak_thres_rel=0.1,
			r_to_sigraw=3,
			pixel_size = 108.4,
			diagnostic=False,
			pltshow=False,
			plot_r=True,
			blob_marker='^',
			blob_markersize=10,
			blob_markercolor=(0,0,1,0.8),
			truth_df=None):

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Prepare blobs_df~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	columns = ['frame', 'x', 'y', 'sig_raw', 'r', 'peak', 'mass', 'mean', 'std']
	blobs_df = pd.DataFrame([], columns=columns)
	plt_array = []

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Update blobs_df~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	for i in range(len(pims_frames)):
		current_frame = pims_frames[i]
		fnum = current_frame.frame_no
		if isinstance(truth_df, pd.DataFrame):
			current_truth_df = truth_df[truth_df['frame'] == fnum]
		else:
			current_truth_df = None

		tmp, tmp_plt_array = detect_blobs(pims_frames[i],
					   min_sig=min_sig,
					   max_sig=max_sig,
					   num_sig=num_sig,
					   blob_thres=blob_thres,
					   peak_thres_rel=peak_thres_rel,
					   r_to_sigraw=r_to_sigraw,
					   pixel_size=pixel_size,
					   diagnostic=diagnostic,
					   pltshow=pltshow,
					   plot_r=plot_r,
					   blob_marker=blob_marker,
					   blob_markersize=blob_markersize,
					   blob_markercolor=blob_markercolor,
					   truth_df=current_truth_df)
		blobs_df = pd.concat([blobs_df, tmp], sort=True)
		plt_array.append(tmp_plt_array)

	blobs_df.index = range(len(blobs_df))
	plt_array = np.array(plt_array)

	return blobs_df, plt_array


def file1_exists_or_pimsopen_file2(head_str, tail_str1, tail_str2):
	if osp.exists(head_str + tail_str1):
		frames = pims.open(head_str + tail_str1)
	else:
		frames = pims.open(head_str + tail_str2)
	return frames


def nonempty_exists_then_copy(input_path, output_path, filename):
	not_empty = len(filename)!=0
	exists_in_input = osp.exists(input_path + filename)
	not_exists_in_output = not osp.exists(output_path + filename)

	if not_empty and exists_in_input and not_exists_in_output:
		frames = imread(input_path + filename)
		frames = frames / frames.max()
		frames = img_as_ubyte(frames)
		imsave(output_path + filename, frames)


def nonempty_openfile1_or_openfile2(path, filename1, filename2):
	if filename1 and osp.exists(path + filename1): # if not empty and exists
		frames = imread(path + filename1)
	else:
		frames = imread(path + filename2)
	return frames


class Pipeline2():

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

		# frames = frames / frames.max()
		# frames = img_as_ubyte(frames)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif', frames)

		if frames.ndim == 3 and self.config.DICT['End frame index'] <= len(frames):
			frames = frames[list(self.config.TRANGE),:,:]

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif', frames)

		print('\tRegi_ref_file: [%s]' % self.config.REF_FILE_NAME)
		print('\tMask_dist2boundary_file: [%s]' % self.config.DIST2BOUNDARY_MASK_NAME)
		print('\tMask_dist253bp1_file: [%s]' % self.config.DIST253BP1_MASK_NAME)
		# load reference files
		nonempty_exists_then_copy(self.config.INPUT_PATH, self.config.OUTPUT_PATH, self.config.REF_FILE_NAME)
		nonempty_exists_then_copy(self.config.INPUT_PATH, self.config.OUTPUT_PATH, self.config.DIST2BOUNDARY_MASK_NAME)
		nonempty_exists_then_copy(self.config.INPUT_PATH, self.config.OUTPUT_PATH, self.config.DIST253BP1_MASK_NAME)


	def split(self):
		split_tif(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-raw.tif', 100)


	def rename(self):
		rename_01(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '.tif')


	def check_regi(self):

		# """
		# ~~~~~~~~~~~~~~~~~change scaler to list~~~~~~~~~~~~~~~~~
		# """

		ref_ind_num = []
		sig_mask = []
		thres_rel = []
		poly_deg = []
		rotation_multiplier =[]
		translation_multiplier = []
		use_ransac = []

		if not isinstance(self.config.REF_IND_NUM, list):
			ref_ind_num.append(self.config.REF_IND_NUM)
			self.config.REF_IND_NUM = ref_ind_num
		if not isinstance(self.config.SIG_MASK, list):
			sig_mask.append(self.config.SIG_MASK)
			self.config.SIG_MASK = sig_mask
		if not isinstance(self.config.THRES_REL, list):
			thres_rel.append(self.config.THRES_REL)
			self.config.THRES_REL = thres_rel
		if not isinstance(self.config.POLY_DEG, list):
			poly_deg.append(self.config.POLY_DEG)
			self.config.POLY_DEG = poly_deg
		if not isinstance(self.config.ROTATION_MULTIPLIER, list):
			rotation_multiplier.append(self.config.ROTATION_MULTIPLIER)
			self.config.ROTATION_MULTIPLIER = rotation_multiplier
		if not isinstance(self.config.TRANSLATION_MULTIPLIER, list):
			translation_multiplier.append(self.config.TRANSLATION_MULTIPLIER)
			self.config.TRANSLATION_MULTIPLIER = translation_multiplier
		if not isinstance(self.config.USE_RANSAC, list):
			use_ransac.append(self.config.USE_RANSAC)
			self.config.USE_RANSAC = use_ransac

		print("######################################")
		print("Check regi parameters")
		print("######################################")

		# If no regi ref file, use raw file automatically
		ref_im = nonempty_openfile1_or_openfile2(self.config.OUTPUT_PATH,
					self.config.REF_FILE_NAME,
					self.config.ROOT_NAME+'-raw.tif')[list(self.config.TRANGE),:,:]

		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif'):
			im = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')
		else:
			im = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME
					+ '-raw.tif')[list(self.config.TRANGE),:,:]

		concate_tif = np.array([])
		# for j in range(len(self.config.REF_IND_NUM)):
		for j in self.config.REF_IND_NUM:
			for k in range(len(self.config.SIG_MASK)):
				for l in range(len(self.config.THRES_REL)):
					for m in range(len(self.config.POLY_DEG)):
						for n in range(len(self.config.ROTATION_MULTIPLIER)):
							for o in range(len(self.config.TRANSLATION_MULTIPLIER)):
								for p in range(len(self.config.USE_RANSAC)):

									# Get regi parameters from ref file, save the regi params in csv file
									regi_params_array_2d = get_regi_params(ref_im,
									              # ref_ind_num=self.config.REF_IND_NUM[j],
												  ref_ind_num=j,
									              sig_mask=self.config.SIG_MASK[k],
									              thres_rel=self.config.THRES_REL[l],
									              poly_deg=self.config.POLY_DEG[m],
									              rotation_multplier=self.config.ROTATION_MULTIPLIER[n],
									              translation_multiplier=self.config.TRANSLATION_MULTIPLIER[o],
									              diagnostic=False)

									# Apply the regi params, save the registered file
									registered = apply_regi_params(im, regi_params_array_2d)

									savename = self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-' \
										+ str(self.config.REF_IND_NUM[j]) + '-' \
										+ str(self.config.SIG_MASK[k]) + '-' \
										+ str(self.config.THRES_REL[l]) + '-' \
										+ str(self.config.POLY_DEG[m]) + '-' \
										+ str(self.config.ROTATION_MULTIPLIER[n]) + '-' \
										+ str(self.config.TRANSLATION_MULTIPLIER[o]) + '-' \
										+ str(self.config.USE_RANSAC[p]) \
										+ '.tif'
									imsave(savename, registered)

									if concate_tif.size != 0:
										concate_tif = np.concatenate((concate_tif, registered), axis=2)
									else:
										concate_tif = registered

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-check-results.tif', concate_tif)
		return


	def regi(self):

		check_regi_switch = isinstance(self.config.REF_IND_NUM, list) or \
							isinstance(self.config.SIG_MASK, list) or \
							isinstance(self.config.THRES_REL, list) or \
							isinstance(self.config.POLY_DEG, list) or \
							isinstance(self.config.ROTATION_MULTIPLIER, list) or \
							isinstance(self.config.TRANSLATION_MULTIPLIER, list) or \
							isinstance(self.config.USE_RANSAC, list)
		if check_regi_switch:
			self.check_regi()
		else:
			print("######################################")
			print("Registering Image Stack")
			print("######################################")

			# If no regi ref file, use raw file automatically
			ref_im = nonempty_openfile1_or_openfile2(self.config.OUTPUT_PATH,
						self.config.REF_FILE_NAME,
						self.config.ROOT_NAME+'-raw.tif')[list(self.config.TRANGE),:,:]

			if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif'):
				im = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')
			else:
				im = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME
						+ '-raw.tif')[list(self.config.TRANGE),:,:]

			# Get regi parameters from ref file, save the regi params in csv file
			regi_params_array_2d = get_regi_params(ref_im,
			              ref_ind_num=self.config.REF_IND_NUM,
			              sig_mask=self.config.SIG_MASK,
			              thres_rel=self.config.THRES_REL,
			              poly_deg=self.config.POLY_DEG,
			              rotation_multplier=self.config.ROTATION_MULTIPLIER,
			              translation_multiplier=self.config.TRANSLATION_MULTIPLIER,
			              diagnostic=True,
						  show_trace=True,
						  use_ransac=self.config.USE_RANSAC)

			# regi_data = pd.DataFrame(regi_params_array_2d,
			# 		columns=['x_center', 'y_center', 'angle', 'delta_x', 'delta_y' ])
			# regi_data.to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME +
			#  				'-regiData.csv', index=False)

			# Apply the regi params, save the registered file
			registered = apply_regi_params(im, regi_params_array_2d)
			imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regi.tif', registered)
			# imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif', registered)


	def get_boundary_mask(self):
		# If no mask ref file, use raw file automatically
		frames = nonempty_openfile1_or_openfile2(self.config.OUTPUT_PATH,
					self.config.DIST2BOUNDARY_MASK_NAME,
					self.config.ROOT_NAME+'-raw.tif')[list(self.config.TRANGE),:,:]

		# If regi params csv file exsits, load it and do the registration.
		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regiData.csv'):
			regi_params_array_2d = pd.read_csv(self.config.OUTPUT_PATH +
			 				self.config.ROOT_NAME + '-regiData.csv').to_numpy()
			frames = apply_regi_params(frames, regi_params_array_2d)

		boundary_masks = get_thres_mask_batch(frames,
					self.config.MASK_SIG_BOUNDARY, self.config.MASK_THRES_BOUNDARY)

		return boundary_masks


	def get_53bp1_mask(self):
		# If no mask ref file, use raw file automatically
		frames = nonempty_openfile1_or_openfile2(self.config.OUTPUT_PATH,
					self.config.DIST253BP1_MASK_NAME,
					self.config.ROOT_NAME+'-raw.tif')[list(self.config.TRANGE),:,:]

		# If regi params csv file exsits, load it and do the registration.
		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regiData.csv'):
			regi_params_array_2d = pd.read_csv(self.config.OUTPUT_PATH +
							self.config.ROOT_NAME + '-regiData.csv').to_numpy()
			frames = apply_regi_params(frames, regi_params_array_2d)

		# Get mask file and save it using 255 and 0
		masks_53bp1 = get_thres_mask_batch(frames,
							self.config.MASK_SIG_53BP1, self.config.MASK_THRES_53BP1)

		return masks_53bp1


	def get_53bp1_blob_mask(self):
		# If no mask ref file, use raw file automatically
		frames = nonempty_openfile1_or_openfile2(self.config.OUTPUT_PATH,
					self.config.MASK_53BP1_BLOB_NAME,
					self.config.ROOT_NAME+'-raw.tif')[list(self.config.TRANGE),:,:]

		# If regi params csv file exsits, load it and do the registration.
		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-regiData.csv'):
			regi_params_array_2d = pd.read_csv(self.config.OUTPUT_PATH +
							self.config.ROOT_NAME + '-regiData.csv').to_numpy()
			frames = apply_regi_params(frames, regi_params_array_2d)

		# Get mask file and save it using 255 and 0
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-tempFile.tif',
				frames)
		pims_frames = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME +
								'-tempFile.tif')
		os.remove(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-tempFile.tif')

		blobs_df, det_plt_array = detect_blobs_batch(pims_frames,
									min_sig=self.config.MASK_53BP1_BLOB_MINSIG,
									max_sig=self.config.MASK_53BP1_BLOB_MAXSIG,
									num_sig=self.config.MASK_53BP1_BLOB_NUMSIG,
									blob_thres=self.config.MASK_53BP1_BLOB_THRES,
									peak_thres_rel=self.config.MASK_53BP1_BLOB_PKTHRES_REL,
									r_to_sigraw=1.4,
									pixel_size=self.config.PIXEL_SIZE,
									diagnostic=False,
									pltshow=False,
									plot_r=False,
									truth_df=None)

		masks_53bp1_blob = blobs_df_to_mask(frames, blobs_df)

		return masks_53bp1_blob


	def mask_boundary(self):
		print("######################################")
		print("Generate mask_boundary")
		print("######################################")
		boundary_masks = self.get_boundary_mask()

		# Save it using 255 and 0
		boundary_masks = np.rint(boundary_masks / \
							boundary_masks.max() * 255).astype(np.uint8)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-boundaryMask.tif',
				boundary_masks)


	def mask_53bp1(self):
		print("######################################")
		print("Generate mask_53bp1")
		print("######################################")
		masks_53bp1 = self.get_53bp1_mask()

		# Save it using 255 and 0
		masks_53bp1 = np.rint(masks_53bp1 / \
							masks_53bp1.max() * 255).astype(np.uint8)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-53bp1Mask.tif',
				masks_53bp1)


	def mask_53bp1_blob(self):
		print("######################################")
		print("Generate mask_53bp1_blob")
		print("######################################")

		masks_53bp1_blob = self.get_53bp1_blob_mask()

		# Save it using 255 and 0
		masks_53bp1_blob = np.rint(masks_53bp1_blob / \
							masks_53bp1_blob.max() * 255).astype(np.uint8)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-53bp1BlobMask.tif',
				masks_53bp1_blob)

	def deno_gaus(self):

		print("######################################")
		print('Applying Gaussian Filter')
		print("######################################")

		frames = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')
		frames = frames / frames.max()
		frames = img_as_ubyte(frames)
		filtered = filter_batch(frames, method='gaussian', arg=self.config.GAUS_BLUR_SIG)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif', filtered)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-deno.tif', filtered)


	def deno_box(self):

		print("######################################")
		print('Applying Boxcar Filter')
		print("######################################")

		frames = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')
		frames = frames / frames.max()
		frames = img_as_ubyte(frames)
		filtered = filter_batch(frames, method='boxcar', arg=self.config.BOXCAR_RADIUS)

		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif', filtered)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-deno.tif', filtered)


	def check_detect_fit(self):

		print("######################################")
		print("Check detection and fitting")
		print("######################################")

		frames = file1_exists_or_pimsopen_file2(self.config.OUTPUT_PATH + self.config.ROOT_NAME,
									'-regi.tif', '-raw.tif')

		frames_deno = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')

		blobs_df, det_plt_array = detect_blobs(frames[0],
									min_sig=self.config.MIN_SIGMA,
									max_sig=self.config.MAX_SIGMA,
									num_sig=self.config.NUM_SIGMA,
									blob_thres=self.config.THRESHOLD,
									peak_thres_rel=self.config.PEAK_THRESH_REL,
									r_to_sigraw=1,
									pixel_size=self.config.PIXEL_SIZE,
									diagnostic=True,
									pltshow=True,
									plot_r=False,
									truth_df=None)

		psf_df, fit_plt_array = fit_psf(frames_deno[0],
		            blobs_df,
		            diagnostic=True,
		            pltshow=True,
		            diag_max_dist_err=self.config.FILTERS['MAX_DIST_ERROR'],
		            diag_max_sig_to_sigraw = self.config.FILTERS['SIG_TO_SIGRAW'],
		            truth_df=None,
		            segm_df=blobs_df)


	def detect_fit(self):

		print("######################################")
		print("Detect, Fit")
		print("######################################")

		frames = file1_exists_or_pimsopen_file2(self.config.OUTPUT_PATH + self.config.ROOT_NAME,
									'-regi.tif', '-raw.tif')

		frames_deno = pims.open(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-active.tif')

		blobs_df, det_plt_array = detect_blobs_batch(frames_deno,
									min_sig=self.config.MIN_SIGMA,
									max_sig=self.config.MAX_SIGMA,
									num_sig=self.config.NUM_SIGMA,
									blob_thres=self.config.THRESHOLD,
									peak_thres_rel=self.config.PEAK_THRESH_REL,
									r_to_sigraw=1,
									pixel_size=self.config.PIXEL_SIZE,
									diagnostic=True,
									pltshow=False,
									plot_r=False,
									truth_df=None)


		det_plt_array = anno_raw(frames_deno, blobs_df)
		imsave(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-detVideo.tif', det_plt_array)

		psf_df, fit_plt_array = fit_psf_batch(frames_deno,
		            blobs_df,
		            diagnostic=False,
		            pltshow=False,
		            diag_max_dist_err=self.config.FILTERS['MAX_DIST_ERROR'],
		            diag_max_sig_to_sigraw = self.config.FILTERS['SIG_TO_SIGRAW'],
		            truth_df=None,
		            segm_df=None)
		psf_df = psf_df.apply(pd.to_numeric)
		psf_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-fittData.csv', index=False)

		self.config.save_config()


	def filt_track(self):

		print("######################################")
		print("Filter and Linking")
		print("######################################")

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

		traj_num_before = blobs_df['particle'].nunique()

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
		blobs_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
									'-physData.csv', index=False)

		after_filter_df = blobs_df [blobs_df['traj_length'] > self.config.FILTERS['TRAJ_LEN_THRES']]
		print("######################################")
		print("Trajectory number before filters: \t%d" % traj_num_before)
		print("Trajectory number after filters: \t%d" % after_filter_df['particle'].nunique())
		print("######################################")


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

					)
		fig.savefig(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-results.pdf', dpi=300)
		# plt.clf(); plt.close()
		plt.show()


	def phys(self):

		print("######################################")
		print("Add Physics Parameters")
		print("######################################")
		blobs_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		# add 'dist_to_boundary' column
		print("######################################")
		print("Add 'dist_to_boundary'")
		print("######################################")
		boundary_masks = self.get_boundary_mask()
		phys_df = add_dist_to_boundary_batch(blobs_df, boundary_masks)

		# add 'dist_to_53bp1' column
		print("######################################")
		print("Add 'dist_to_53bp1'")
		print("######################################")
		masks_53bp1 = self.get_53bp1_mask()
		phys_df = add_dist_to_53bp1_batch(blobs_df, masks_53bp1)

		# Save '-physData.csv'
		phys_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)


	def phys_dist2boundary(self):
		print("######################################")
		print("Add Physics Param: dist_to_boundary")
		print("######################################")
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')
		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-boundaryMask.tif'):
			boundary_masks = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-boundaryMask.tif')
			boundary_masks = boundary_masks // 255
		else:
			boundary_masks = self.get_boundary_mask()
		phys_df = add_dist_to_boundary_batch(phys_df, boundary_masks)
		phys_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)


	def phys_dist253bp1(self):
		print("######################################")
		print("Add Physics Param: dist_to_53bp1")
		print("######################################")
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')
		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-53bp1Mask.tif'):
			masks_53bp1 = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-53bp1Mask.tif')
			masks_53bp1 = masks_53bp1 // 255
		else:
			masks_53bp1 = self.get_53bp1_mask()
		phys_df = add_dist_to_53bp1_batch(phys_df, masks_53bp1)
		phys_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)


	def phys_dist253bp1_blob(self):
		print("######################################")
		print("Add Physics Param: dist_to_53bp1_blob")
		print("######################################")
		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')
		if osp.exists(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-53bp1BlobMask.tif'):
			masks_53bp1_blob = imread(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-53bp1BlobMask.tif')
			masks_53bp1_blob = masks_53bp1_blob // 255
		else:
			masks_53bp1_blob = self.get_53bp1_blob_mask()
		phys_df = add_dist_to_53bp1_batch(phys_df, masks_53bp1_blob)
		phys_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)


	def sort_plot(self):

		print("######################################")
		print("Sort and PlotMSD")
		print("######################################")

		frames = file1_exists_or_pimsopen_file2(self.config.OUTPUT_PATH + self.config.ROOT_NAME,
									'-regi.tif', '-raw.tif')

		phys_df = pd.read_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + '-physData.csv')

		if self.config.DO_SORT & \
			('dist_to_53bp1' in phys_df.columns) & \
			('dist_to_boundary' in phys_df):
			phys_df = sort_phys(phys_df, self.config.SORTERS)
			phys_df.round(6).to_csv(self.config.OUTPUT_PATH + self.config.ROOT_NAME + \
						'-physData.csv', index=False)
		else:
			sorter_list = get_sorter_list(phys_df)
			phys_df = phys_df.drop(columns=sorter_list[1:-1])

		# Apply traj_length_thres filter
		phys_df = phys_df[ phys_df['traj_length'] > self.config.FILTERS['TRAJ_LEN_THRES'] ]

		plot_msd_batch(phys_df,
					 image=frames[0],
					 output_path=self.config.OUTPUT_PATH,
					 root_name=self.config.ROOT_NAME,
					 pixel_size=self.config.PIXEL_SIZE,
					 frame_rate=self.config.FRAME_RATE,
					 divide_num=self.config.DIVIDE_NUM,
					 plot_without_sorter=False,
					 show_fig=False,
					 save_pdf=True,
					 open_pdf=False)

		self.config.DICT['Load existing analMeta'] = True
		self.config.save_config()

	def merge_plot(self):

		start_ind = self.config.ROOT_NAME.find('_')
		end_ind = self.config.ROOT_NAME.find('_', start_ind+1)
		today = str(date.today().strftime("%y%m%d"))
		merged_name = today + self.config.ROOT_NAME[start_ind:end_ind]

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
			print(phys_files)

			if len(phys_files) > 1:
				phys_df = merge_physdfs(phys_files)
				phys_df = relabel_particles(phys_df)
			else:
				phys_df = pd.read_csv(phys_files[0])

			phys_df.round(6).to_csv(self.config.OUTPUT_PATH + merged_name + \
							'-physDataMerged.csv', index=False)

		# Apply traj_length_thres filter
		phys_df = phys_df[ phys_df['traj_length'] > self.config.FILTERS['TRAJ_LEN_THRES'] ]

		# phys_df = phys_df.loc[phys_df['exp_label'] == 'BLM']
		fig = plot_merged(phys_df, 'exp_label',
						pixel_size=self.config.PIXEL_SIZE,
						frame_rate=self.config.FRAME_RATE,
						divide_num=self.config.DIVIDE_NUM,
						RGBA_alpha=0.5,
						do_gmm=False)

		fig.savefig(self.config.OUTPUT_PATH + merged_name + '-mergedResults.pdf')


def get_root_name_list(settings_dict):
	# Make a copy of settings_dict
	# Use '*%#@)9_@*#@_@' to substitute if the labels are empty
	settings = settings_dict.copy()
	if settings['Regi reference file label'] == '':
		settings['Regi reference file label'] = '*%#@)9_@*#@_@'
	if settings['Mask boundary_mask file label'] == '':
		settings['Mask boundary_mask file label'] = '*%#@)9_@*#@_@'
	if settings['Mask 53bp1_mask file label'] == '':
		settings['Mask 53bp1_mask file label'] = '*%#@)9_@*#@_@'
	if settings['Mask 53bp1_blob_mask file label'] == '':
		settings['Mask 53bp1_blob_mask file label'] = '*%#@)9_@*#@_@'

	root_name_list = []
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
			if (settings['Mask boundary_mask file label'] not in temp+'.tif') and \
				(settings['Mask 53bp1_mask file label'] not in temp+'.tif') and \
				(settings['Regi reference file label'] not in temp+'.tif') and \
				(settings['Mask 53bp1_blob_mask file label'] not in temp+'.tif'):
				root_name_list.append(temp)

	return np.array(sorted(root_name_list))


def analMeta_to_dict(analMeta_path):
	df = pd.read_csv(analMeta_path, header=None, index_col=0, na_filter=False)
	df = df.rename(columns={1:'value'})
	srs = df['value']

	dict = {}
	for key in srs.index:
		try: dict[key] = ast.literal_eval(srs[key])
		except: dict[key] = srs[key]
	return dict


def pipeline_batch(settings_dict, control_list):

	# """
	# ~~~~~~~~~~~~~~~~~1. Get root_name_list~~~~~~~~~~~~~~~~~
	# """
	root_name_list = get_root_name_list(settings_dict)

	print("######################################")
	print("Data to be processed:")
	print(root_name_list)
	print("######################################")

	for root_name in root_name_list:

		print("Processing: %s" % root_name)

		# """
		# ~~~~~~~~~~~~~~~~~2. Update config~~~~~~~~~~~~~~~~~
		# """

		config = Config(settings_dict)

		# 2.0. If LOAD_ANALMETA==True, then load existing analMeta file, if there is one
		if config.DICT['Load existing analMeta'] & \
		osp.exists(settings_dict['IO input_path'] + root_name + '-analMeta.csv'):
			existing_settings = analMeta_to_dict(settings_dict['IO input_path'] + root_name + '-analMeta.csv')
			existing_settings['IO input_path']= settings_dict['IO input_path']
			existing_settings['IO output_path'] = settings_dict['IO output_path']
			existing_settings['Processed By:'] = settings_dict['Processed By:']
			existing_settings.pop('Processed by:', None)
			settings_dict = existing_settings
			config = Config(settings_dict)

		# 2.1. Update config.ROOT_NAME and config.DICT
		config.ROOT_NAME = root_name
		config.DICT['Raw data file'] = root_name + '.tif'
		config.DICT['Processed date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		config.DICT['Processed by:'] = settings_dict['Processed By:']

		# 2.2. Update config.REF_FILE_NAME
		if '-' in root_name and root_name.find('-')>0:
			key = root_name[0:root_name.find('-')]
		else:
			key = root_name

		if settings_dict['Regi reference file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' + key +
					'*' + settings_dict['Regi reference file label'] + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.REF_FILE_NAME = file_list[0].split('/')[-1]
			else:
				config.DICT['Regi reference file label'] = ''

		# 2.3. Update config.DIST2BOUNDARY_MASK_NAME
		if settings_dict['Mask boundary_mask file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' + key +
					'*' + settings_dict['Mask boundary_mask file label'] + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.DIST2BOUNDARY_MASK_NAME = file_list[0].split('/')[-1]
			else:
				config.DICT['Mask boundary_mask file label'] = ''

		# 2.4. Update config.DIST253BP1_MASK_NAME
		if settings_dict['Mask 53bp1_mask file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' + key +
					'*' + settings_dict['Mask 53bp1_mask file label'] + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.DIST253BP1_MASK_NAME = file_list[0].split('/')[-1]
			else:
				config.DICT['Mask 53bp1_mask file label'] = ''

		# 2.5. Update config.MASK_53BP1_BLOB_NAME
		if settings_dict['Mask 53bp1_blob_mask file label']:# if label is not empty, find file_list
			file_list = np.array(sorted(glob(settings_dict['IO input_path'] + '*' + key +
					'*' + settings_dict['Mask 53bp1_blob_mask file label'] + '*')))
			if len(file_list) == 1: # there should be only 1 file targeted
				config.MASK_53BP1_BLOB_NAME = file_list[0].split('/')[-1]
			else:
				config.DICT['Mask 53bp1_blob_mask file label'] = ''

		# """
		# ~~~~~~~~~~~~~~~~~3. Setup pipe and run~~~~~~~~~~~~~~~~~
		# """
		pipe = Pipeline2(config)
		for func in control_list:
			getattr(pipe, func)()
