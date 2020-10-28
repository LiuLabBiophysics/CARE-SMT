import numpy as np

def clr_code_mask(mask, int_df, typ_arr, typ_clr_arr=None):

	"""

	Pseudo code
	----------
	1. Initialize RGB mask
	2. Get unique labels for each cell type
	3. Color code the RGB mask

	Parameters
	----------
	mask : ndarray,
		Mask to be transformed to color
	int_df: DataFrame,
		Intensity DataFrame
	typ_arr: list,
		List of different region types e.g. [type1, type2, ...]
	typ_clr_rr: list,
		List of colors to use for each type.

	Returns
	-------
	mask_rgb: ndarray,
		Color coded mask

	"""

	tmp = np.zeros_like(mask)
	mask_rgb = np.dstack((tmp, tmp, tmp))

	for i, typ in enumerate(typ_arr):
		lbl_by_type = int_df.loc[int_df['cell_type'] == typ, 'label'].unique()
		for lbl in lbl_by_type:
			typ_mask = np.where(mask == lbl)
			mask_rgb[:,:][typ_mask] = typ_clr_arr[i]

	return mask_rgb
