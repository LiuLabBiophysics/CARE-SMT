import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, clear_border, mark_boundaries
from skimage.measure import label, regionprops_table
from ..util import *

def find_nuclei(image,
				min_distance=1,
				seed_thres_rel=0.6,
				mask=None,
				pltshow=False):

	"""

	Finds nuclei in an image by thresholding, applying distance transform
	to a binary mask, and detecting local maxima in the distance image.
	Note: this function does not perform segmentation.

	Parameters
	----------

	image : ndarray,
		a single frame or stack of frames
	min_distance: int, optional
		the minimum distance separating two nuclei
	threshold_rel : float, optional
		relative threshold between 0 and 1
	pltshow: list, optional
		whether or not to show a validation figure

	Returns
	-------
	blobs_df: DataFrame,
		a DataFrame containing coordinates of seeds for a segmentation routine
	dist_map_arr: ndarray,
		an ndarray containing distance maps

	"""

	# """
	# ~~~~~~~~~~~Initialize data structs~~~~~~~~~~~~~
	# """

	columns = ['frame', 'x', 'y']
	blobs_df = pd.DataFrame([], columns=columns)
	dist_map_arr = np.zeros_like(image)

	# """
	# ~~~~~~~~~~~Find nuclei~~~~~~~~~~~~~
	# """

	nframes = len(image)
	for i in range(nframes):

		print('Finding cells in frame %d/%d' % (i, nframes))

		# """
		# ~~~~~~~~~~~Threshold~~~~~~~~~~~~~
		# """

		if mask:
			binary = mask
		else:
			thresh = threshold_otsu(image[i])
			binary = image[i] > thresh

		dist_map = ndi.distance_transform_edt(binary)
		dist_map = gaussian(dist_map)

		coordinates = peak_local_max(dist_map,
									 min_distance=min_distance,
									 threshold_rel=seed_thres_rel,
									 indices=True)

		# """
		# ~~~~~~~~~~~Scatter detections~~~~~~~~~~~~~
		# """

		if pltshow:
			plt.imshow(image[i], cmap='gray')
			plt.scatter(coordinates[:, 1],
						coordinates[:, 0],
						marker='^', s=7,
						color='black')
			plt.tight_layout(); plt.show()

		# """
		# ~~~~~~~~~~~Scatter detections~~~~~~~~~~~~~
		# """

		this_blob_df = pd.DataFrame(data=coordinates, columns=['x','y'])
		this_blob_df = this_blob_df.assign(frame=i)
		blobs_df = pd.concat([blobs_df, this_blob_df])
		dist_map_arr[i] = dist_map

	blobs_df = blobs_df.assign(r=1)


	return blobs_df, dist_map_arr


def do_watershed(image,
				 min_distance=1,
				 seed_thres_rel=0.6,
				 mask=None,
				 pltshow=False):

	"""

	Performs a classic watershed segmentation by thresholding, finding seeds
	from a distance map, and flooding

	Parameters
	----------

	image : ndarray,
		a single frame or stack of frames
	min_distance: int, optional
		the minimum distance separating two nuclei
	threshold_rel : float, optional
		relative threshold between 0 and 1
	pltshow: list, optional
		whether or not to show a validation figure

	Returns
	-------
	blobs_df: DataFrame,
		DataFrame containing coordinates, labels, and object areas
	mask_arr: ndarray,
		an ndarray of object masks

	"""

	# """
	# ~~~~~~~~~~Reshape the image if 2D~~~~~~~~~~~~
	# """

	mask_arr = np.zeros_like(image)

	# """
	# ~~~~~~~~~~Find watershed seeds~~~~~~~~~~~~
	# """

	seed_df, dist_map_arr = find_nuclei(image,
										min_distance=min_distance,
	 					    			seed_thres_rel=seed_thres_rel,
										mask=mask,
										pltshow=False)

	# """
	# ~~~~~~~~~~Perform watershed segmentation~~~~~~~~~~~~
	# """

	nframes = len(image)
	for i in range(nframes):
		print('Segmenting frame %d/%d' % (i, nframes))

		this_df = seed_df.loc[seed_df['frame'] == i]
		dist_map = dist_map_arr[i]; binary = dist_map > 0
		coordinates = this_df[['x','y']].to_numpy()
		seed_arr = np.zeros_like(image[i])
		for coordinate in coordinates:
			seed_arr[int(coordinate[0]), int(coordinate[1])] = 255

		seed_arr = ndi.label(seed_arr)[0]
		mask = watershed(-dist_map, seed_arr, mask=binary)
		mask = clear_border(mask)
		mask_arr[i] = mask

		if pltshow:
			fig, ax = plt.subplots(ncols=3, figsize=(9, 3),
								   sharex=True, sharey=True)

			marked = mark_boundaries(image[i], mask)
			ax[0].imshow(marked, cmap=plt.cm.gray)
			ax[0].scatter(coordinates[:, 1], coordinates[:, 0], color='red')
			ax[2].imshow(mask, cmap='coolwarm')
			ax[1].imshow(dist_map, cmap=plt.cm.nipy_spectral)
			for a in ax:
				a.set_axis_off()
			plt.show()

	return mask_arr

def track_masks(mask_arr,
				memory=5,
				z_filter=0.5,
				search_range=20,
				min_traj_length=10,
				min_size=None,
				do_filter=False):

	"""

	Builds a DataFrame from a movie of masks and runs the
	tracking algorithm on that DataFrame

	Parameters
	----------
	mask_arr : ndarray
		ndarray containing the masks

	search_range: int
		the maximum distance the centroid of a mask can move between frames
		and still be tracked

	memory: int
		the number of frames to remember a mask that has disappeared


	Returns
	-------

	"""

	# """
	# ~~~~~~~~~~~Extract mask_df from mask_arr~~~~~~~~~~~~~~
	# """

	mask_df = pd.DataFrame([])
	nframes = mask_arr.shape[0]

	for i in range(nframes):

		props = regionprops_table(mask_arr[i], properties=('centroid', 'area', 'label'))
		this_df = pd.DataFrame(props).assign(frame=i)
		this_df = this_df.rename(columns={'centroid-0':'x', 'centroid-1':'y'})
		mask_df = pd.concat([mask_df, this_df])

	# """
	# ~~~~~~~~~~~Link Trajectories~~~~~~~~~~~~~~
	# """

	mask_df = tp.link_df(mask_df, search_range=search_range, memory=memory)
	mask_df = mask_df.reset_index(drop=True)

	# """
	# ~~~~~~~~~~~Apply filters~~~~~~~~~~~~~~
	# """

	if do_filter:

		print("######################################")
		print("Filtering out suspicious data points")
		print("######################################")

		grp = mask_df.groupby('particle')['area']
		mask_df['z_score'] = grp.apply(lambda x: np.abs((x - x.mean()))/x.std())
		mask_df = mask_df.loc[mask_df['z_score'] < z_filter]

		if min_size:
			mask_df = mask_df.loc[mask_df['area'] > min_size]

		mask_df = tp.link_df(mask_df, search_range=search_range, memory=memory)
		mask_df = tp.filter_stubs(mask_df, min_traj_length)
		mask_df = mask_df.reset_index(drop=True)

	# """
	# ~~~~~~~~~~~Check if DataFrame is empty~~~~~~~~~~~~~
	# """

	if mask_df.empty:
		print('\n***Trajectories num is zero***\n')
		return

	return mask_df
