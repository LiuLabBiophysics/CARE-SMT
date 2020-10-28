import numpy as np; import pandas as pd; import pims
from ..plot.plotutil import anno_blob, anno_scatter
from ..plot.plotutil import plot_end
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from ..deno import median
from ..io import imshow_gray
from skimage.util import img_as_ubyte
from scipy.ndimage import gaussian_laplace
from skimage.filters.thresholding import _cross_entropy
from skimage.util import img_as_float
from skimage.feature import peak_local_max
from skimage.filters import gaussian

def detect_blobs(pims_frame,

				min_sig=1,
				max_sig=3,
				num_sig=5,
				blob_thres_rel=0,
				overlap=0.5,
				peak_min=0,
				num_peaks=10,
				peak_thres_rel=0,
				mass_thres_rel=0,
				peak_r_rel=0,
				mass_r_rel=0,

				r_to_sigraw=3,
				pixel_size =0.1084,

				diagnostic=True,
				pltshow=True,
				plot_r=True,
				blob_marker='^',
				blob_markersize=10,
				blob_markercolor=(0,0,1,0.8),
				truth_df=None):
	"""
	Detect blobs for each frame.

	Parameters
	----------
	pims_frame : pims.Frame object
		Each frame in the format of pims.Frame.
	min_sig : float, optional
		As 'min_sigma' argument for blob_log().
	max_sig : float, optional
		As 'max_sigma' argument for blob_log().
	num_sig : int, optional
		As 'num_sigma' argument for blob_log().
	blob_thres_rel : float, optional
		Relative level in log space to determine
		the 'threshold' argument for blob_log().
	peak_thres_rel : float, optional
		Relative peak threshold [0,1].
		Blobs below this relative value are removed.
	mass_thres_rel : float, optional
		Relative mass threshold [0,1].
		Blobs below this relative value are removed.
	peak_r_rel : float, optional
		Relative peak times r threshold [0,1].
		Blobs below this relative value are removed.
	mass_r_rel : float, optional
		Relative mass times r threshold [0,1].
		Blobs below this relative value are removed.
	r_to_sigraw : float, optional
		Multiplier to sigraw to decide the fitting patch radius.
	pixel_size : float, optional
		Pixel size in um. Used for the scale bar.
	diagnostic : bool, optional
		If true, run the diagnostic.
	pltshow : bool, optional
		If true, show diagnostic plot.
	plot_r : bool, optional
		If True, plot the blob boundary.
	truth_df : DataFrame or None. optional
		If provided, plot the ground truth position of the blob.

	Returns
	-------
	blobs_df : DataFrame
		columns = ['frame', 'x', 'y', 'sig_raw', 'r',
					'peak', 'mass', 'mean', 'std']
	plt_array :  ndarray
		ndarray of diagnostic plot.

	Examples
	--------
	import pims
	from cellquantifier.smt.detect import detect_blobs, detect_blobs_batch
	frames = pims.open('cellquantifier/data/simulated_cell.tif')
	detect_blobs(frames[0])
	"""

	# """
	# ~~~~~~~~~~~~~~~~~Detection using skimage.feature.blob_log~~~~~~~~~~~~~~~~~
	# """

	frame = pims_frame

	# threshold automation
	frame_f = img_as_float(frame)
	frame_log = -gaussian_laplace(frame_f, sigma=min_sig) * min_sig**2

	maxima = peak_local_max(frame_log,
				threshold_abs=0,
				footprint=None,
				num_peaks=num_peaks)

	columns = ['x', 'y', 'peak_log',]
	maxima_df = pd.DataFrame([], columns=columns)
	maxima_df['x'] = maxima[:, 0]
	maxima_df['y'] = maxima[:, 1]
	maxima_df['peak_log'] = frame_log[ maxima_df['x'], maxima_df['y'] ]
	if blob_thres_rel=='auto':
		blob_thres_final = maxima_df['peak_log'].mean()*0.01
	else:
		blob_thres_final = maxima_df['peak_log'].mean()*blob_thres_rel

	# peak_thres_rel automation
	maxima = peak_local_max(frame,
				threshold_abs=0,
				footprint=None,
				num_peaks=num_peaks)

	columns = ['x', 'y', 'peak',]
	maxima_df = pd.DataFrame([], columns=columns)
	maxima_df['x'] = maxima[:, 0]
	maxima_df['y'] = maxima[:, 1]
	maxima_df['peak'] = frame[ maxima_df['x'], maxima_df['y'] ]
	if peak_thres_rel=='auto':
		peak_thres_abs = maxima_df['peak'].mean()*0.05
	else:
		peak_thres_abs = maxima_df['peak'].max() * peak_thres_rel

	blobs = blob_log(frame,
					 min_sigma=min_sig,
					 max_sigma=max_sig,
					 num_sigma=num_sig,
					 threshold=blob_thres_final,
					 overlap=overlap,
					 )

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~Prepare blobs_df and update it~~~~~~~~~~~~~~~~~~~~~~
	# """

	columns = ['frame', 'x', 'y', 'sig_raw', 'r', 'peak', 'mass']
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
		blobs_df.at[i, 'mass'] = blob.sum()

	# """
	# ~~~~~~~Filter detections~~~~~~~
	# """
	mass_thres_abs = blobs_df['mass'].max()*mass_thres_rel

	r_max = blobs_df['r'].max()
	r_min = blobs_df['r'].min()

	pk_max = blobs_df['peak'].max()
	pk_min = blobs_df['peak'].min()
	# slope_pk = (pk_max - pk_min) / (r_min - r_max)
	# intersect_pk = pk_max - slope_pk * r_min
	
	mass_max = blobs_df['mass'].max()
	mass_min = blobs_df['mass'].min()
	# slope_mass = (mass_max - mass_min) / (r_min - r_max)
	# intersect_mass = mass_max - slope_mass * r_min


	blobs_df_nofilter = blobs_df.copy()
	blobs_df = blobs_df[ (blobs_df['peak'] >= peak_min) ]
	blobs_df = blobs_df[ (blobs_df['peak'] > peak_thres_abs) ]
	blobs_df = blobs_df[ (blobs_df['mass'] > mass_thres_abs) ]
	# blobs_df = blobs_df[ blobs_df['peak'] > \
	# 				(blobs_df['r']*slope_pk+intersect_pk)*peak_r_rel ]
	# blobs_df = blobs_df[ blobs_df['mass'] > \
	# 				(blobs_df['r']*slope_mass+intersect_mass)*mass_r_rel ]

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
		fig, ax = plt.subplots(2, 2, figsize=(12,12))

		# """
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~Annotate the blobs~~~~~~~~~~~~~~~~~~~~~~~~~~
		# """
		ax[0][0].imshow(frame, cmap="gray", aspect='equal')
		anno_blob(ax[0][0], blobs_df_nofilter, marker=blob_marker, markersize=blob_markersize,
				plot_r=plot_r, color=blob_markercolor)
		ax[0][0].text(0.95,
				0.05,
				"Foci_num before filter: %d" %(len(blobs_df_nofilter)),
				horizontalalignment='right',
				verticalalignment='bottom',
				fontsize = 12,
				color = (0.5, 0.5, 0.5, 0.5),
				transform=ax[0][0].transAxes,
				weight = 'bold',
				)

		ax[0][1].imshow(frame, cmap="gray", aspect='equal')
		anno_blob(ax[0][1], blobs_df, marker=blob_marker, markersize=blob_markersize,
				plot_r=plot_r, color=blob_markercolor)
		ax[0][1].text(0.95,
				0.05,
				"Foci_num after filter: %d" %(len(blobs_df)),
				horizontalalignment='right',
				verticalalignment='bottom',
				fontsize = 12,
				color = (0.5, 0.5, 0.5, 0.5),
				transform=ax[0][1].transAxes,
				weight = 'bold',
				)

		# """
		# ~~~~~~~~~~~~~~~~~~~Annotate ground truth if needed~~~~~~~~~~~~~~~~~~~
		# """
		if isinstance(truth_df, pd.DataFrame):
			anno_scatter(ax[0][0], truth_df, marker='o', color=(0,1,0,0.8))

		# """
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Add scale bar~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# """
		font = {'family': 'arial', 'weight': 'bold','size': 16}
		scalebar = ScaleBar(pixel_size, 'um', location = 'upper right',
			font_properties=font, box_color = 'black', color='white')
		scalebar.length_fraction = .3
		scalebar.height_fraction = .025
		ax[0][0].add_artist(scalebar)

		# """
		# ~~~~Plot foci in parameter space~~~~
		# """
		x2, y2 = blobs_df_nofilter['r'], blobs_df_nofilter['peak']
		ax[1][0].scatter(x2, y2, marker='^', c=[(0,0,1)])

		x3, y3 = blobs_df_nofilter['r'], blobs_df_nofilter['mass']
		ax[1][1].scatter(x3, y3, marker='^', c=[(0,0,1)])

		delta_sig = (max_sig-min_sig) * 0.1
		x2_thres = np.linspace(min_sig-delta_sig, max_sig+delta_sig, 50) * r_to_sigraw
		# y2_thres = (slope_pk * x2_thres + intersect_pk) * peak_r_rel
		y2_peak = x2_thres / x2_thres * peak_thres_abs
		# ax[1][0].plot(x2_thres, y2_thres, '--', c=(0,0,0,0.8), linewidth=3)
		ax[1][0].plot(x2_thres, y2_peak, '--', c=(0,0,0,0.8), linewidth=3)
		ax[1][0].set_xlabel('r')
		ax[1][0].set_ylabel('peak')
		delta_pk = (pk_max-pk_min)*0.1
		ax[1][0].set_ylim(0, pk_max+delta_pk)

		x3_thres = np.linspace(min_sig-delta_sig, max_sig+delta_sig, 50) * r_to_sigraw
		# y3_thres = (slope_mass * x3_thres + intersect_mass) * mass_r_rel
		y3_peak = x3_thres / x3_thres * mass_thres_abs
		# ax[1][1].plot(x3_thres, y3_thres, '--', c=(0,0,0,0.8), linewidth=3)
		ax[1][1].plot(x3_thres, y3_peak, '--', c=(0,0,0,0.8), linewidth=3)
		ax[1][1].set_xlabel('r')
		ax[1][1].set_ylabel('mass')
		delta_mass = (mass_max-mass_min)*0.1
		ax[1][1].set_ylim(0, mass_max+delta_mass)

		plt_array = plot_end(fig, pltshow)

	return blobs_df, plt_array


def detect_blobs_batch(pims_frames,
			min_sig=1,
			max_sig=3,
			num_sig=5,
			blob_thres_rel=0.1,
			overlap=0.5,
			peak_min=0,
			num_peaks=10,
			peak_thres_rel=0.1,
			mass_thres_rel=0,
			peak_r_rel=0,
			mass_r_rel=0,
			r_to_sigraw=3,
			pixel_size = 108.4,
			diagnostic=False,
			pltshow=False,
			plot_r=True,
			blob_marker='^',
			blob_markersize=10,
			blob_markercolor=(0,0,1,0.8),
			truth_df=None):

	"""
	Detect blobs for the whole movie.

	Parameters
	----------
	See detect_blobs().

	Returns
	-------
	blobs_df : DataFrame
		columns = ['frame', 'x', 'y', 'sig_raw', 'r',
					'peak', 'mass', 'mean', 'std']
	plt_array :  ndarray
		ndarray of diagnostic plots.

	Examples
	--------
	import pims
	from cellquantifier.smt.detect import detect_blobs, detect_blobs_batch
	frames = pims.open('cellquantifier/data/simulated_cell.tif')
	detect_blobs_batch(frames, diagnostic=0)
	"""

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
					   blob_thres_rel=blob_thres_rel,
					   overlap=overlap,
					   peak_min=peak_min,
					   num_peaks=num_peaks,
					   peak_thres_rel=peak_thres_rel,
					   mass_thres_rel=mass_thres_rel,
					   peak_r_rel=peak_r_rel,
					   mass_r_rel=mass_r_rel,
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
