import pandas as pd
import re

def relabel_particles(df, col1='raw_data', col2='particle'):

	"""
	Relabel particles after merging dataframes from several experiments

	Pseudocode
	----------
	1. For each unique file get all particles
	2. For each particle reassign label to particle counter (i)

	Parameters
	----------
	df: DataFrame

	Returns
	-------
	df: DataFrame
		input df with relabeled particles

	"""

	df.sort_values(by=[col1, col2])
	file_names = df[col1].unique()
	i = 0

	ind = 1
	tot = len(file_names)
	for file_name in file_names:
		print("Relabeling (%d/%d): %s" % (ind, tot, file_name))
		ind = ind + 1

		sub_df = df.loc[df[col1] == file_name]
		particles = sub_df[col2].unique()

		for particle in particles:
			df.loc[(df[col1] == file_name) & \
			(df[col2] == particle), 'tmp'] = i
			i+=1

	df['tmp'] = df['tmp'].astype('int')
	df[col2] = df['tmp']; del df['tmp']

	return df

def merge_physdfs(files, mode='basic'):

	"""
	Relabel particles after merging dataframes from several experiments

	Parameters
	----------
	files: list
	    list of physData files to be merged

	Returns
	-------
	merged_df: DataFrame
	    DataFrame after merging

	"""
	temp_df = pd.read_csv(files[0], index_col=False)
	columns = temp_df.columns.tolist()
	merged_df = pd.DataFrame([], columns=columns)

	ind = 1
	tot = len(files)
	for file in files:
		print("Merging (%d/%d): %s" % (ind, tot, file))
		ind = ind + 1

		df = pd.read_csv(file, index_col=False)

		# add 'rat_data' column to the merged df
		root_name = file.split('/')[-1]
		df = df.assign(raw_data=root_name)

		# add 'exp_label' column to the merged df
		if mode=='basic':
			exp = re.findall(r'[a-zA-Z]{3}\d{1}', file)
			df = df.assign(exp_label=exp[0][:-1])

		if mode=='general':
		    if 'cohort' in root_name:
		        df = df.assign(exp_label=root_name[0:8])
		    else:
		        m = root_name.find('_') + 1
		        n = root_name.find('_', m)
		        df = df.assign(exp_label=root_name[m:n])

		if mode=='mengdi':
			m = root_name.find('_') + 1
			m = root_name.find('_', m) + 1
			n = root_name.find('-', m)
			df = df.assign(exp_label=root_name[m:n])

		if mode=='stiffness':
			m = root_name.find('-') + 1
			m = root_name.find('-', m) + 1
			n = root_name.find('_') + 1
			n = root_name.find('_', n)
			df = df.assign(exp_label=root_name[m:n])

		merged_df = pd.concat([merged_df, df], sort=True, ignore_index=True)

	return merged_df

def merge_physdfs2(files):

    """
    Used for cycled 53bp1 imaging experiments. Merges physData.csv files
    from all cycles and all cells at one time

    Parameters
    ----------
    files: list
        list of physData files to be merged

    Returns
    -------
    merged_df: DataFrame
        DataFrame after merging

    """

    temp_df = pd.read_csv(files[0], index_col=False)
    columns = temp_df.columns.tolist()
    merged_df = pd.DataFrame([], columns=columns)

    for file in files:
        df = pd.read_csv(file, index_col=False)

        # add 'rat_data' column to the merged df
        root_name = file.split('/')[-1]
        df = df.assign(raw_data=root_name)

        # add 'exp_label' column to the merged df
        cell_num = ''.join(re.findall("cell\d{2}", file))
        exp = file.split('_')[1]
        exp = ''.join(re.findall("[a-zA-Z]+", exp))

        df = df.assign(exp_label=exp)
        df = df.assign(cell_num=cell_num)

        merged_df = pd.concat([merged_df, df], sort=True, ignore_index=True)

    return merged_df

def add_cycle_col(files):

    """
    Add cycle column for cycled imaging scenarios

    Parameters
    ----------
    files: list
        list of physData files to be modified

    """

    for file in files:
        chunks = file.split('/')
        folder = ''.join(chunks[:-1])
        filename = chunks[-1]
        df = pd.read_csv(file, index_col=False)
        if 'cycle_num' not in df:
            # add 'cycle_num' column to the merged df
            str = filename.split('_')[1]
            cycle_num = int(str.split('-')[-1])
            df = df.assign(cycle_num=cycle_num)
            df.to_csv(file)
