from glob import glob

def get_files_from_prefix(input_dir, prefix, tags=None):

	"""
	Pseudo code
	----------
	1. If tags are specified, find all files containg prefix and any of the tags
	2. Unpack nested array

	Parameters
	----------
	input_dir : str,
		Directory containing files of interest

	prefix : str,
		Prefix in filename to filter files by

	tags : list,
		Identifiers of channels to use to find files

	"""

	if tags:
		fn_arr = [glob(input_dir + prefix + '*%s*' % tag) \
				  for tag in tags]
		fn_arr = [fn for sub_fn_arr in fn_arr for fn in sub_fn_arr]

	else:
		fn_arr = glob(input_dir + prefix + '*')

	return fn_arr

def get_unique_prefixes(input_dir, tag=None):

	"""
	Pseudo code
	----------
	1. If tag is specified, get all files in input_dir that contain tag
	2. Extract unique prefixes from the list of files

	Parameters
	----------
	input_dir : str,
		Directory containing files of interest

	tag : str,
		Identifier for files of interest


	"""

	if tag:
		files = glob(input_dir + '*%s*' % (tag))
	else:
		files = glob(input_dir + '*')

	roots = [file.split('/')[-1] for file in files]
	prefixes = ['_'.join(root.split('_')[:-1]) for root in roots]
	prefixes = list(set(prefixes))

	return prefixes
