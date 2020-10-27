# import numpy as np
# import math
# from skimage.util import img_as_ubyte
#
# def filter(image, method, arg):
#
# 	"""Dispatches denoise request for a single frame to appropriate function
#
# 	Parameters
# 	----------
# 	image : 2D ndarray
#
# 	method : string
# 		the method to use when denoising the image
#
# 	arg: int or float
# 		the argument passed to gaussian(), gain(), or boxcar() functions
#
# 	Returns
# 	-------
# 	denoised : 2D ndarray
# 		The denoise image
# 	"""
#
# 	if method == 'gaussian':
# 		denoised[i] = gaussian(image, arg)
# 	elif method == 'gain':
# 		denoised[i] = gain(image, arg)
# 	elif method == 'boxcar':
# 		denoised[i] = boxcar(image, arg)
#
# 	return denoised
#
# def filter_batch(image, method, arg):
#
# 	"""Dispatches denoise request for a time-series to appropriate function
#
# 	Parameters
# 	----------
# 	image : 2D ndarray
#
# 	method : string
# 		The method to use when denoising the image
#
# 	Returns
# 	-------
# 	denoised : 3D ndarray
# 		The denoised image
# 	"""
#
# 	denoised = np.zeros(image.shape, dtype='uint8')
#
# 	ind = 1
# 	tot = len(image)
# 	for i in range(len(image)):
# 		print("Denoising (%d/%d)" % (ind, tot))
# 		ind = ind + 1
#
# 		if method == 'gaussian':
# 			denoised[i] = gaussian(image[i], arg)
#
# 		elif method == 'gain':
# 			denoised[i] = gain(image[i], arg)
#
# 		elif method == 'boxcar':
# 			denoised[i] = boxcar(image[i], arg)
#
# 		elif method == 'mean':
# 			denoised[i] = mean(image[i], arg)
#
# 		elif method == 'median':
# 			denoised[i] = median(image[i], arg)
#
# 		elif method == 'minimum':
# 			denoised[i] = minimum(image[i], arg)
#
# 	return denoised
#
# def gaussian(image, sigma):
#
# 	"""Wrapper for skimage.filters.gaussian
#
#     Parameters
#     ----------
#     image : ndarray
#
#     sigma : int
#         standard deviation of gaussian kernel used to convolve the image
#
#     Returns
#     -------
#     blurred : ndarray
#         The image after gaussian blurring
#     """
#
# 	from skimage.filters import gaussian
#
# 	image = image / image.max()
# 	image = img_as_ubyte(image)
#
# 	image = gaussian(image, sigma)
#
# 	image = image / image.max()
# 	image = img_as_ubyte(image)
#
# 	return image
#
#
# def gain(image, gain):
#
# 	"""Introduce image gain via scalar multiplication
#
#     Parameters
#     ----------
#     image : ndarray
#
#     gain : int
#         multiplicative factor
#
#     Returns
#     -------
#     image_gain : ndarray
#         The image after scalar multiplication
#     """
#
# 	image = img_as_ubyte(image*gain)
#
# 	return image
#
# def boxcar(image, width):
#
# 	"""Subtract background using boxcar convolution
#
#     Parameters
#     ----------
#     image : ndarray
#
#     width : int
#         half width (radius of the boxcar kernel)
#
#     Returns
#     -------
#     filtered : ndarray
#         The image after background subtraction
#     """
#
# 	from scipy import signal
# 	from skimage import img_as_ubyte
#
# 	def normalize(array):
#
# 		array = [x/sum(array) for x in array]
# 		return array
#
# 	def zero(filtered_image_array, width, lnoise):
#
# 		filtered = filtered_image_array
# 		count = 0
#
# 		lzero = int(max(width,math.ceil(5*lnoise)))
# 		#TOP
# 		for row in range(0, lzero):
# 			for element in range(0, len(filtered[row])):
# 				filtered[row, element] = 0
# 		#BOTTOM
# 		for row in range(-lzero, -0):
# 			for element in range(0, len(filtered[row])):
# 				filtered[row, element] = 0
# 		#LEFT
# 		for row in range(0, len(filtered)):
# 			for element_index in range(0, lzero):
# 				filtered[row, element_index] = 0
# 		#RIGHT
# 		for row in range(0, len(filtered)):
# 			for element_index in range(-lzero, -0):
# 				filtered[row, element_index] = 0
# 		#set negative pixels in filtered to 0
# 		for row in range(0, len(filtered)):
# 			for element_index in range(0, len(filtered[row])):
# 				if filtered[row, element_index] < 0:
# 					count +=1
# 					filtered[row, element_index] = 0
#
# 		return filtered
#
# 	#normalize image
# 	image = image/255
#
# 	#build boxcar kernel
# 	length = len(range(-1*int(round(width)), int(round(width)) + 1))
# 	boxcar_kernel = [float(1) for i in range(0, length)]
# 	boxcar_kernel = normalize(boxcar_kernel)
#
# 	#convert 1d to 2d
# 	boxcar_kernel = np.reshape(boxcar_kernel, (1, len(boxcar_kernel)))
# 	filtered = signal.convolve2d(image.transpose(),boxcar_kernel.transpose(),'same')
# 	filtered = signal.convolve2d(filtered.transpose(), boxcar_kernel.transpose(),'same')
# 	filtered = img_as_ubyte(image-filtered)
#
# 	return filtered
#
#
# def mean(image, disk_radius):
# 	from skimage.morphology import disk
# 	from skimage.filters import rank
#
# 	image = image / image.max()
# 	image = img_as_ubyte(image)
#
# 	selem = disk(disk_radius)
# 	image = rank.mean(image, selem=selem)
#
# 	image = image / image.max()
# 	image = img_as_ubyte(image)
#
# 	return image
#
#
# def median(image, disk_radius):
# 	from skimage.morphology import disk
# 	from skimage.filters import median
#
# 	image = image / image.max()
# 	image = img_as_ubyte(image)
#
# 	selem = disk(disk_radius)
# 	image = median(image, selem=selem)
#
# 	image = image / image.max()
# 	image = img_as_ubyte(image)
#
# 	return image
#
#
# def minimum(image, disk_radius):
# 	from skimage.morphology import disk, erosion, dilation
#
# 	image = image / image.max()
# 	image = img_as_ubyte(image)
#
# 	selem = disk(disk_radius)
# 	image = erosion(image, selem=selem)
# 	image = dilation(image, selem=selem)
#
# 	image = image / image.max()
# 	image = img_as_ubyte(image)
#
# 	return image
