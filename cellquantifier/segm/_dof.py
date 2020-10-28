import cv2
from skimage.measure import regionprops

def dof(im, winh=10, winw=10, stride=10):

	"""Calculates the degree of focus (DOF) of an image over
       local regions by using a sliding window approach and the
       variance of laplacian operator

	Parameters
	----------
	im : 2d/3d ndarray
	winh : float, optional
		Height of sliding window
	winhw : float, optional
		Width of sliding window
	stride : int, optional
		Stride for sliding window. Set to winw for non-overlapping windows


	Returns
	-------
	dof_im : 2d/3d ndarray
		The degree of focus image

	"""

	def sliding_window(im, stride, win_size):
		for x in range(0, im.shape[0], stride):
			for y in range(0, im.shape[1], stride):
				yield (x, y, im[y:y + win_size[1], x:x + win_size[0]])


	if len(im.shape) == 2:
		im = im.reshape((1,) + im.shape)

	dof_im = np.zeros_like(im)
	win = sliding_window(im, stride=stride, win_size=(winw, winh))
    for i in range(len(im)):
    	for (x, y, window) in win:
    		dof = np.std(cv2.Laplacian(window, cv2.CV_64F)) ** 2
    		dof_im[y:y+winw, x:x+winh] = dof

	return dof_im

def dof_mask(mask, dof_im):

	"""Calculates the average degree of focus (DOF) over masked
       regions, replacing object labels with the average

	Parameters
	----------
	mask : 2d/3d ndarray

	dof_im : float, optional
		Height of sliding window

	Returns
	-------
	dof_im : 2d/3d ndarray
		The degree of focus mask

	"""

    dof_im[binary == 0] = 0
	props = regionprops(mask, dof_im)
	for prop in props:
		mask[mask == prop.label] = prop.mean_intensity

    return dof_im
