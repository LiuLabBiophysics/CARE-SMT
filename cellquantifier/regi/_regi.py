import numpy as np
import matplotlib.pyplot as plt
from cellquantifier.plot.plotutil import format_ax
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from skimage.filters import gaussian
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
from skimage.transform import SimilarityTransform, warp
from ..qmath.ransac import ransac_polyfit
from ..plot.plotutil import anno_ellipse
from ..segm.mask import get_thres_mask as get_mask


def get_regi_params(array_3d,
			  ref_ind_num=0,
			  sig_mask=3,
			  thres_rel=0.2,
			  poly_deg=2,
			  rotation_multplier=1,
			  translation_multiplier=1,
			  diagnostic=False,
			  show_trace=True,
			  use_ransac=False,
			  frame_rate=3.33):
	"""
	Get the rigid registration parameters.

	Parameters
	----------
	array_3d : ndarray
		3d ndarray from tif file.
	ref_ind_num : int, optional
		reference index number.
	sig_mask : float, optional
		Sigma for mask generation.
	thres_rel : float, optional
		Relative threshold for mask generation.
	poly_deg : int, optional
		Polynomial degree.
	rotation_multplier : float, optional
		Use this number to manually control the rotation amplitude.
	translation_multplier : float, optional
		Use this number to manually control the translation amplitude.
	diagnostic : bool, optional
		If True, run the diagnostic.

	Returns
	-------
	regi_params_array_2d: ndarray
		Registration parameters for each frame.
		Five columns: 'x0', 'y0', 'angle', 'delta_x', 'delta_y'.
	"""

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~Get Ref_mask and Ref_props~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	Ref = array_3d[ref_ind_num]
	Ref_mask = get_mask(Ref, sig=sig_mask, thres_rel=thres_rel)
	Ref_props = regionprops(Ref_mask)

	# """
	# ~~~~~~~~~~~~~~~~Get registration parameters for each frame~~~~~~~~~~~~~~~~
	# """

	regi_params_array_2d = np.zeros((len(array_3d), 5))
	for i in range(len(array_3d)):
		Reg = array_3d[i]
		Reg_mask = get_mask(Reg, sig=sig_mask, thres_rel=thres_rel)
		Reg_props = regionprops(Reg_mask)

		# params for rotation
		angle = (Ref_props[0].orientation - Reg_props[0].orientation) / np.pi * 180
		center = (Reg_props[0].centroid[1], Reg_props[0].centroid[0])

		# params for translation
		delta_r = Reg_props[0].centroid[0] - Ref_props[0].centroid[0]
		delta_c = Reg_props[0].centroid[1] - Ref_props[0].centroid[1]
		delta_y, delta_x = delta_r, delta_c

		regi_params_array_2d[i] = np.array([center[0], center[1],
											angle, delta_x, delta_y])

		print("Get params NO.%d is done!" % i)

	# """
	# ~~~~~~~~~~~~~~~~Ploynomial fitting the rotation~~~~~~~~~~~~~~~~
	# """

	index = range(len(regi_params_array_2d))

	angle_raw = np.array(regi_params_array_2d[:,2])
	poly_params1 = np.polyfit(index, angle_raw, poly_deg)
	poly_params2 = ransac_polyfit(index, angle_raw, poly_deg,
				min_sample_num=len(index) // 2,
				residual_thres=0.1,
				max_trials=300)
	p1 = np.poly1d(poly_params1)
	p2 = np.poly1d(poly_params2)
	angle_fit1 = p1(index) * rotation_multplier
	angle_fit2 = p2(index) * rotation_multplier
	if use_ransac:
		regi_params_array_2d[:, 2] = np.array(angle_fit2)
	else:
		regi_params_array_2d[:, 2] = np.array(angle_fit1)

	regi_params_array_2d[:, 3] = regi_params_array_2d[:,3] * translation_multiplier
	regi_params_array_2d[:, 4] = regi_params_array_2d[:,4] * translation_multiplier

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Diagnostic~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	if diagnostic:
		fig, ax = plt.subplots(1, 2, figsize=(12, 6))
		index = np.array(index)/frame_rate
		ax[0].plot(index, angle_raw, '.-r', label='RAW')
		ax[0].plot(index, angle_fit1, '-k', label='POLY')
		ax[0].plot(index, angle_fit2, '-b', label='RANSAC')
		ax[0].legend(loc='upper right')

		if show_trace:
			Ref = adjust_gamma(Ref, gain=1)
			trace = mark_boundaries(Ref, Reg_mask)
			ax[1].imshow(trace, cmap='gray')
		else:
			ax[1].imshow(Reg_mask, cmap='gray')

		anno_ellipse(ax[1], Ref_props)
		anno_ellipse(ax[1], Reg_props)

		format_ax(ax[0],
				  xlabel=r'$\mathbf{Time (s)}$',
				  ylabel=r'$\mathbf{\Delta\theta (rad)}$',
				  label_fontsize=15,
				  label_fontweight='bold',
				  tklabel_fontsize=13,
				  tklabel_fontweight='bold',
				  ax_is_box=True)

		ax[1].set_xticks([])
		ax[1].set_yticks([])
		plt.show()

	return regi_params_array_2d

def apply_regi_params(array_3d, regi_params_array_2d):
	"""
	Apply rigid registration parameters.

	Parameters
	----------
	array_3d : ndarray
		3d ndarray from tif file.
	regi_params_array_2d: ndarray
		Registration parameters for each frame.
		Five columns: 'x0', 'y0', 'angle', 'delta_x', 'delta_y'.

	Returns
	-------
	tif_array_3d: ndarray
		3d ndarray after registration.

	Examples
	--------
	from skimage.io import imread, imsave
	from cellquantifier.regi._regi import get_regi_params, apply_regi_params
	m=0; delta=100
	tif = imread('53bp1.tif')[m:m+delta]
	params = get_regi_params(tif,
						ref_ind_num=0, sig_mask=3, thres_rel=0.2,
						rotation_multplier=1, poly_deg=1,
						diagnostic=1)
	tif_array_3d = apply_regi_params(tif, params)
	imsave('test_regied.tif', tif_array_3d)
	"""

	tif_array_3d = np.zeros_like(array_3d, dtype=np.uint8)
	for i in range(len(array_3d)):

		# """
		# ~~~~~~~~~~~~~~~~regi the rotation~~~~~~~~~~~~~~~~
		# """
		center = (regi_params_array_2d[i][0], regi_params_array_2d[i][1])
		angle = (regi_params_array_2d[i][2])
		warped1 = rotate(array_3d[i], center=center, angle=angle,
						order=1, preserve_range=True)

		# """
		# ~~~~~~~~~~~~~~~~regi the translation~~~~~~~~~~~~~~~~
		# """
		translation = (regi_params_array_2d[i][3], regi_params_array_2d[i][4])
		tform = SimilarityTransform(translation=translation)
		warped2 = warp(warped1, tform, order=1, preserve_range=True)
		warped2 = np.rint(warped2 / warped2.max() * 255).astype(np.uint8)

		# """
		# ~~~~~~~~~~~~~~~~update tif_array_3d~~~~~~~~~~~~~~~~
		# """
		tif_array_3d[i] = warped2
		print("Registration NO.%d is done!" % i)

	return tif_array_3d
