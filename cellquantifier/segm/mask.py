from skimage.filters import gaussian
import numpy as np
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.filters import threshold_li
from skimage.segmentation import clear_border

def get_li_mask(img, sig=2):

    """
    Get a mask based on gaussian blur, li threshold (automatic)

    Parameters
    ----------
    img : ndarray
        Imaging in the format of 2d ndarray.
    sig : float
        Sigma of gaussian blur

    Returns
    -------
    mask_array_2d: ndarray
        2d ndarray of 0s and 1s
    """
    img = gaussian(img, sigma=sig)
    th = threshold_li(img)
    mask_array_2d = img > .9*th
    mask_array_2d = clear_border(mask_array_2d)
    mask_array_2d = mask_array_2d.astype(np.uint8)

    return mask_array_2d

def get_li_mask_batch(tif, sig=2):
    shape = (len(tif), tif[0].shape[0], tif[0].shape[1])
    masks_array_3d = np.zeros(shape, dtype=np.uint8)
    for i in range(len(tif)):
        masks_array_3d[i] = get_li_mask(tif[i], sig=sig)

    return masks_array_3d

def get_thres_mask(img, sig=3, thres_rel=0.2):
    """
    Get a mask based on "gaussian blur" and "threshold".

    Parameters
    ----------
    img : ndarray
        Imaging in the format of 2d ndarray.
    sig : float
        Sigma of gaussian blur
    thres_rel : float, optional
        Relative threshold comparing to the peak of the image.

    Returns
    -------
    mask_array_2d: ndarray
        2d ndarray of 0s and 1s
    """

    img = gaussian(img, sigma=sig)
    img = img > img.max()*thres_rel
    mask_array_2d = img.astype(np.uint8)

    return mask_array_2d


def get_thres_mask_batch(tif, sig=3, thres_rel=0.2):
    """
    Get a mask stack based on "gaussian blur" and "threshold".

    Parameters
    ----------
    tif : 3darray
		3d ndarray.
    sig : float
        Sigma of gaussian blur
    thres_rel : float, optional
        Relative threshold comparing to the peak of the image.

    Returns
    -------
    masks_array_3d: 3darray
        3d ndarray of 0s and 1s.

    Examples
    --------
    import pims
    from cellquantifier.io import imshow
    from cellquantifier.segm import get_thres_mask_batch
    frames = pims.open('cellquantifier/data/simulated_cell.tif')
    masks = get_thres_mask_batch(frames, sig=3, thres_rel=0.1)
    imshow(masks[0], masks[10], masks[20], masks[30], masks[40])
    """

    shape = (len(tif), tif[0].shape[0], tif[0].shape[1])
    masks_array_3d = np.zeros(shape, dtype=np.uint8)
    for i in range(len(tif)):
        masks_array_3d[i] = get_thres_mask(tif[i], sig=sig, thres_rel=thres_rel)
        # print("Get thres_mask NO.%d is done!" % i)
    print("Get thres_mask is done!")

    return masks_array_3d


def get_dist2boundary_mask(mask, step_size=3):
    """
    Create dist2boundary mask via binary dilation and erosion

	Parameters
	----------
	mask : 2D binary ndarray

	Returns
	-------
	dist_mask : 2D int ndarray
		The distance t0 boundary mask
        if on the mask boundary, the value equals 0.
        if outside of the mask boundary, the value is positive.
        if inside of the mask boundary, the value is negative.

    Examples
	--------
	from cellquantifier.segm.mask import get_thres_mask, get_dist2boundary_mask
    from cellquantifier.io.imshow import imshow
    from cellquantifier.data import simulated_cell

    m = 285; n = 260; delta = 30
    img = simulated_cell()[0][m:m+delta, n:n+delta]
    mask = get_thres_mask(img, sig=1, thres_rel=0.5)
    dist2boundary_mask = get_dist2boundary_mask(mask, step_size=5)
    imshow(dist2boundary_mask)
	"""

    dist_mask = np.zeros(mask.shape, dtype=int)
    selem = disk(step_size)
    dist_mask[ mask==0 ] = 999999
    dist_mask[ mask==1 ] = -999999

    mask_outwards = mask.copy()
    i = 1
    while True:
        dilated_mask_outwards = binary_dilation(mask_outwards, selem=selem)
        dist_mask[ dilated_mask_outwards ^ mask_outwards==1 ] = i * step_size
        mask_outwards = dilated_mask_outwards
        i = i + 1
        if np.count_nonzero(dist_mask == 999999) == 0:
            break

    mask_inwards = mask.copy()
    i = 0
    while True:
        shrinked_mask_inwards = binary_erosion(mask_inwards, selem=selem)
        dist_mask[ shrinked_mask_inwards ^ mask_inwards==1 ] = i * step_size
        mask_inwards = shrinked_mask_inwards
        i = i - 1
        if np.count_nonzero(dist_mask == -999999) == 0:
            break

    return dist_mask


def get_dist2boundary_mask_batch(masks, step_size=3):
    """
    Create dist2boundary mask via binary dilation and erosion

	Parameters
	----------
	masks : 3D binary ndarray

	Returns
	-------
	dist_masks : 3D int ndarray

    Examples
	--------
    from skimage.io import imread
    from cellquantifier.io import imshow
    from cellquantifier.segm import (get_thres_mask_batch,
                                    get_dist2boundary_mask_batch)
    tif = imread('cellquantifier/data/simulated_cell.tif')[:, 285:385, 260:360]
    masks = get_thres_mask_batch(tif, sig=1, thres_rel=0.5)
    dist_masks = get_dist2boundary_mask_batch(masks, step_size=5)
    imshow(tif[0], tif[10], tif[20], tif[30], tif[40])
    imshow(masks[0], masks[10], masks[20], masks[30], masks[40])
    imshow(dist_masks[0], dist_masks[10], dist_masks[20], dist_masks[30],
            dist_masks[40])
	"""

    shape = (len(masks), masks[0].shape[0], masks[0].shape[1])
    dist_masks = np.zeros(shape, dtype=int)
    for i in range(len(masks)):
        dist_masks[i] = get_dist2boundary_mask(masks[i], step_size=step_size)
        print("Get distance_mask NO.%d is done!" % i)

    return dist_masks


def blobs_df_to_mask(tif, blobs_df):

    shape = (len(tif), tif[0].shape[0], tif[0].shape[1])
    masks_array_3d = np.zeros(shape, dtype=np.uint8)

    for i in blobs_df['frame'].unique():
        curr_blobs_df = blobs_df[ blobs_df['frame']==i ]
        mask = np.zeros((shape[1], shape[2]))
        x = np.arange(0, shape[1])[:, np.newaxis]
        y = np.arange(0, shape[2])[np.newaxis, :]

        for j in curr_blobs_df.index:
            x0 = curr_blobs_df.loc[j, 'x']
            y0 = curr_blobs_df.loc[j, 'y']
            r = curr_blobs_df.loc[j, 'r']
            curr_mask = (x-x0)**2 + (y-y0)**2 <= r**2
            mask = np.logical_or(mask, curr_mask)

        masks_array_3d[i][mask] = 1
        print("Get blob_mask NO.%d is done!" % i)

    return masks_array_3d
