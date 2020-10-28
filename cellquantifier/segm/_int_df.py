import pandas as pd
from skimage.measure import regionprops

def get_int_df(im_arr, mask, im_names=['ch1', 'ch2']):

	"""
	Pseudo code
	----------
	1. Extract region properties from each intensity image
	2. Build intensity dataframe from region properties

	Parameters
	----------

	im_arr : list,
		List of intensity images

	mask : ndarray,
		Mask with optionally labeled regions

	im_names : list,
		Identifiers of channels to use in output DataFrame

	"""

	df_arr = []
	for i, im in enumerate(im_arr):
		prop = regionprops(mask, intensity_image=im)
		prop = [[p.label, p.mean_intensity] for p in prop]
		df = pd.DataFrame(prop, columns=['label','avg_%s_intensity' \
													% (im_names[i])])
		df_arr.append(df)


	int_df = pd.concat(df_arr, axis=1)
	int_df = int_df.loc[:,~int_df.columns.duplicated()]

	return int_df
