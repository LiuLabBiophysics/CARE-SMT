import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from scipy import stats
from skimage.measure import label


def add_obj_lbl(df, mask, is_labeled=True, col_name='region_label'):

	"""
	Add object_lbl column to DataFrame

	Parameters
	----------
	df : DataFrame
		DataFrame with x,y columns

	"""

	if not is_labeled:
		mask = label(mask)

	for i, row in df.iterrows():
		df.at[i, col_name] = int(mask[int(round(df.at[i, 'x'])), \
									    int(round(df.at[i, 'y']))])
	return df

def add_obj_type(mask_arr, blobs_df, col_name='region_type',
				 obj_names=None, pltshow=False, det_colors=None):

	"""

	Adds a obj type column to blobs_df. Each unique obj could be
	cytoplasm, nucleus, etc.

	Pseudo code
	----------
	1. Create region type mask
	2. Call add_obj_lbl() where region labels are encoded region types
	3. Convert encoded region types to preferred names

	Parameters
	----------

	mask_arr : list,
		list of masks each defining a unique object
	blobs_df: DataFrame
		dataframe containing detection instances and coordinates
	col_name : str, optional
		name to use for region type column
	obj_names: list, optional
		names to use for different regions (in same order as mask_arr)

	Example
	----------
	#read statements
	blobs_df = add_obj_type([parent_mask, child_mask], blobs_df=blobs_df,
							 obj_names=['cyto', 'nuc'])

	"""

	# """
	# ~~~~~~~~~~~Create the region type mask~~~~~~~~~~~~~~
	# """

	reg_type_mask = np.zeros_like(mask_arr[0])

	for i, mask in enumerate(mask_arr):
		reg_type_mask[mask > 0] = i + 1

	# """
	# ~~~~~~~~~~~Add the numerical labels from reg_type_mask~~~~~~~~~~~~~~
	# """

	blobs_df = add_obj_lbl(blobs_df, reg_type_mask, col_name=col_name)
	num_types = sorted(blobs_df['region_type'].unique())


	# """
	# ~~~~~~~~~~~Check if user has supplied detection colors~~~~~~~~~~~~~~
	# """

	if not det_colors:
		det_colors = mpl.cm.coolwarm(np.linspace(0,1,len(num_types)))

	# """
	# ~~~~~~~~~~~Show the reg_type_mask and detections~~~~~~~~~~~~~~
	# """

	if pltshow:
		plt.imshow(reg_type_mask, cmap='gray')
		for i, type in enumerate(num_types):
			this_df = blobs_df.loc[blobs_df['region_type'] == type]
			plt.scatter(this_df['y'],this_df['x'], color=det_colors[i], s=2)
		plt.show()

	# """
	# ~~~~~~~~~~~Convert numerical labels to names, if specified~~~~~~~~~~~~~~
	# """


	if len(obj_names) != len(num_types):
		obj_names.insert(0, 'bg')

	if obj_names:
		for i, num_type in enumerate(num_types):
			blobs_df.loc[blobs_df['region_type'] == num_type, \
						'region_type'] = obj_names[i]

	return blobs_df

def bin_df(df, bin_col, nbins=10):

	"""
	Add 'category' column to dataframe by binning w.r.t bin_col

	Parameters
	----------

	df : DataFrame
		DataFrame

	bin_col : str,
		column in df to bin by

	nbins : int,
		number of bins

	"""

	df['category'] = pd.cut(df[bin_col], nbins)

	r_min = df[bin_col].to_numpy().min()
	r_max = df[bin_col].to_numpy().max()
	bin_size = (r_max-r_min)/nbins
	hist, bin_edges = np.histogram(df[bin_col], nbins) #get bin edges
	bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1]) #get bin centers

	return bin_centers, df

def norm_df(df, col, col_arr, row_props, nest_col=None):

	"""
	Normalize column(s) of a dataframe based on the value of a particular column
	at a particular row


	Pseudo code
	----------

	0. Divide the dataframe based on unique values of a column (if desired)
	1. Extract the denominator for normalization
	2. Divide column(s) by normalization denominator

	Parameters
	----------

	col : array_like,
		column(s) in DataFrame to be normalized

	row_props : dict,
		dictionary of conditions each key being a column and each value
		being the value of that column

	Example
	----------
	"""

	if nest_col:
		df_arr = nest_df(df, cat_cols=[nest_col])

	norm_df_arr = []
	for this_df in df_arr:

		tmp_df = this_df[row_props.keys()]
		mask = tmp_df.isin(row_props)
		tmp_df = tmp_df[mask].dropna()
		row = tmp_df.index.to_numpy()[0]

		norm_denom = this_df.at[row, col]
		this_df[col_arr] = this_df[col_arr]/norm_denom
		norm_df_arr.append(this_df)

	df = pd.concat(norm_df_arr)

	return df


def nest_df(df, cat_cols, count=-1, data_col=None):

	"""
	Divide a dataframe based on an array of categorical columns, recursively
	The recursion counter is used to index an array of categorical columns.
	Each level of the nested array comes from a different category

	Pseudo code
	----------
	1. Check if the max recursion depth has been met
	2. Increment recursion counter
	2. Iterate over dataframes and replace them with arrays of dataframes

	Parameters
	----------
	ax : object
		matplotlib axis.

	df : DataFrame
		DataFrame that contains cols

	cols : list,
		a list of columns, each of which is a category to divide the data on

	Example
	----------
	"""

	depth = len(cat_cols) - 1

	def do_nest(df_arr, cat_cols, depth, count=count, data_col=data_col):

		if count == depth:
			if data_col:
				return [list(df[data_col].to_numpy())[0] for df in df_arr]
			else:
				return df_arr

		count+=1
		for i, df in enumerate(df_arr):
			df_arr[i] = do_nest([df.loc[df[cat_cols[count]] == cat]
							for cat in df[cat_cols[count]].unique()],
							cat_cols, depth, count, data_col)

		return df_arr

	df = df.sort_values(cat_cols)
	df_arr = do_nest([df], cat_cols, depth, count=count, data_col=data_col)
	df_arr = df_arr[0]

	return df_arr

def get_pval_df(df, data_col, cat_col):

	"""
	Get a correlation-style matrix of p-values on data_col by dividing df
	by cat_col. Useful when there are several unique labels
	for a single category (cat_col)

	Parameters
	----------

	df : DataFrame
		DataFrame containing data_col and cat_col columns
	data_col : str,
		column that should be used to calculate the p-value i.e.
		the variable of statistical interest
	cat_col : str,
		column that should be used for dividing up the dataframe.

	"""

	p_val_arr = []
	cats = df[cat_col].unique()
	cats_df = pd.DataFrame(columns=cats)
	p_val_df = cats_df.transpose().join(cats_df, how='outer')

	for cat1 in cats_df.columns:
		for cat2 in cats_df.columns:
			df1 = df.loc[df[cat_col]==cat1]
			df2 = df.loc[df[cat_col]==cat2]
			t_test = stats.ttest_ind(df1[data_col], df2[data_col])
			p_val_df[cat1][cat2] = t_test[1]

	return p_val_df

def get_binary_pval(df, cat_col, data_col):

	"""

	Calculates the p-value for a pair of labels for a particular category
	e.g. the diffusion coefficient for particles inside and outside the nucleus

	Pseudo code
	----------


	Parameters
	----------
	ax : object
		matplotlib axis.

	df : DataFrame
		DataFrame that contains cols

	cols : list,
		a list of columns, each of which is a category to divide the data on

	Example
	----------
	"""

	if isinstance(df, list):
		return []

	try:
		cat1, cat2 = df[cat_col].unique()
		df1 = df.loc[df[cat_col]==cat1]
		df2 = df.loc[df[cat_col]==cat2]
		t_test = stats.ttest_ind(df1[data_col], df2[data_col])
		return t_test[1]
	except:
		print("The category '%s' is not binary" % str(cat_col))
		return

def df_to_pval(df_arr, cat_col, data_col):

	"""

	Searches recursively for DataFrames and calculates the p-value
	for each DataFrame found by dividing it on cat_col. Useful when
	the category is binary. This function reduces to the simplest
	(non-nested) case as well.

	Pseudo code
	----------


	Parameters
	----------
	ax : object
		matplotlib axis.

	df : DataFrame
		DataFrame that contains cols

	cols : list,
		a list of columns, each of which is a category to divide the data on

	Example
	----------
	"""

	def recurse(df_arr, cat_col, data_col):

		#termination statement
		if not isinstance(df_arr, list):
			p_val = get_binary_pval(df_arr, cat_col, data_col)
			return p_val

		p_val_arr = [recurse(arr, cat_col, data_col) for arr in df_arr]

		return p_val_arr

	if not isinstance(df_arr, list):
		df_arr = [df]

	p_val_arr = recurse(df_arr, cat_col=cat_col,
						data_col=data_col)
	return p_val_arr

def get_frac_df(df, frac_col, group_cols, map_df=None, map_col=None):

	"""

	Transforms a DataFrame by calculating the fraction of records with
	a particular set of column values.

	Pseudocode
	----------
	1. Group by group_cols
	2. Count the number of elements in each group, filling with zeros
	   when necessary
	3. Restore a column lost by grouping via 'mapping', if desired
	4. Compute the fraction dataframe


	Parameters
	----------

	df : DataFrame
		DataFrame containing 'label', 'prefix', 'region_type', and 'count' columns

	group_cols : list,
		columns to group the dataframe by. the size of each group is used

	map_df : dataFrame, optional
		two-column dataframe that maps a column in group_cols to a column
		lost by grouping

	map_col : str, optional
		column that should be used for dividing up the dataframe.

	"""

	# """
	# ~~~~~~~~~~~Group by and get size of groups~~~~~~~~~~~~~
	# """

	df_grp = df.groupby(group_cols)
	df = df_grp.size().unstack(fill_value=0).stack() #fill with zero
	df = df.reset_index(name="count")

	# """
	# ~~~~~~~~~~~Map after group_cols, if desired~~~~~~~~~~~~~~
	# """

	if map_df is not None:
		try:
			for i, row in map_df.iterrows():
				label = row['label']; value = row[map_col]
				df.loc[df['label'] == label, map_col] = value
		except:
			print('Check map_df format (see doc for help)')
			return


	# """
	# ~~~~~~~~~~~Transform to fraction dataframe~~~~~~~~~~~~~
	# """

	frac_df = pd.DataFrame(columns=df.columns)

	for value in df[frac_col].unique():

		this_df = df.loc[df[frac_col] == value]
		counts = this_df['count'].to_numpy()
		this_df['fracs'] = counts/sum(counts)
		frac_df = frac_df.append(this_df)

	del frac_df['count']

	return frac_df

def merge_rna_dfs(input_dir, prefixes):

	"""
	Used for rna imaging experiments (not time-lapse compatible)

	Parameters
	----------
	files: list
		list of physData files to be merged

	Returns
	-------
	merged_df: DataFrame
		DataFrame after merging

	"""

	merged_blobs_df = pd.DataFrame(columns=[])
	merged_int_df = pd.DataFrame(columns=[])

	for prefix in prefixes:

		str = input_dir + prefix
		blobs_df = pd.read_csv(str  + '_hla-fittData-lbld.csv', index_col=0)
		int_df = pd.read_csv(str + '_ins-gluc-intensity-lbld.csv')
		blobs_df = blobs_df.rename(columns={'region_label': 'label'})

		# """
		# ~~~~~~~~~~~Make sure there is only one frame~~~~~~~~~~~~~~
		# """

		if len(blobs_df['frame'].unique()) > 1:
			blobs_df = blobs_df.loc[blobs_df['frame'] == 0]

		# """
		# ~~~~~~~~~~~Assign cell type from int_df to detections~~~~~~~~~~~~~~
		# """

		labels = int_df['label'].unique()
		for label in labels:
			cell_type = int_df.loc[int_df['label'] == label, \
								   'cell_type'].to_numpy()[0]
			blobs_df.loc[blobs_df['label'] == label, 'cell_type'] = cell_type

		# """
		# ~~~~~~~~~~~Filter detections in cells not in int_df~~~~~~~~~~~~~~
		# """

		blobs_df['cell_type'].replace('', np.nan, inplace=True)
		blobs_df = blobs_df.dropna()

		# """
		# ~~~~~~~~~~~Assign prefix and concatenate~~~~~~~~~~~~~~
		# """

		blobs_df = blobs_df.assign(prefix=prefix)
		int_df = int_df.assign(prefix=prefix)

		merged_blobs_df = pd.concat([merged_blobs_df, blobs_df], axis=0, \
									ignore_index=True)
		merged_int_df = pd.concat([merged_int_df, int_df], axis=0, \
		 							ignore_index=True)

	return merged_blobs_df, merged_int_df
