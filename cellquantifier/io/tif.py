from skimage.io import imread, imsave

def split_tif(tif_path, interval_num):
	"""
    Split one tif file into several sub_tif files

    Parameters
    ----------
    tif_path : string
    interval_num : integer

    Returns
    -------
    Several tif files based on interval_num

	Examples (Split a 1100-frames-long tif into 11 100-frames-long tif files)
    --------
	split_tif(tif_path, 100)
    """

	frames = imread(tif_path)
	start_ind = 0
	end_ind = start_ind + interval_num

	if frames.ndim == 3:
		while end_ind <= len(frames):
			save_path = tif_path[:-4] + '-' + str(start_ind) + '-' + str(end_ind) + '.tif'
			imsave(save_path, frames[start_ind:end_ind])
			start_ind = end_ind
			end_ind = start_ind + interval_num
