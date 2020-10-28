import pandas as pd

def sort_phys(df, sorters=None):
	"""
	Label rows with boolean depending on if they meet the sorting criteria

	Parameters
	----------
	df : DataFrame
		DataFrame with column 'particle', 'dist_to_boundary', 'dist_to_53bp1'

	sorters: dict
		dictionary of filters

	Returns
	-------
	sorted_df : DataFrame
		DataFrame after boolean labeling

	Examples
	--------
	import pandas as pd
	from cellquantifier.phys import sort_phys

	sorters = {
		'DIST_TO_BOUNDARY': [-100, 0],
		'DIST_TO_53BP1' : [-5, 0],
	}
	df = pd.read_csv('cellquantifier/data/simulated_cell-physData.csv')
	sorted_df = sort_phys(df, sorters=sorters)
	"""

	sorted_df = df.copy()
	particles = sorted_df['particle'].unique()

	# """
	# ~~~~~~~~~~~~~~~~~Sort dist_to_boundary~~~~~~~~~~~~~~~~~
	# """

	if sorters['DIST_TO_BOUNDARY'] != None:
		avg_dist = sorted_df.groupby('particle')['dist_to_boundary'].mean()
		for particle in particles:
			bool = avg_dist[particle] >= sorters['DIST_TO_BOUNDARY'][0] \
			and avg_dist[particle] <= sorters['DIST_TO_BOUNDARY'][1]
			sorted_df.loc[sorted_df['particle'] == particle, 'sort_flag_boundary'] = bool

	# """
	# ~~~~~~~~~~~~~~~~~Sort dist_to_53bp1~~~~~~~~~~~~~~~~~
	# """

	if sorters['DIST_TO_53BP1'] != None:
		avg_dist = sorted_df.groupby('particle')['dist_to_53bp1'].mean()
		for particle in particles:
			bool = avg_dist[particle] >= sorters['DIST_TO_53BP1'][0] \
			and avg_dist[particle] <= sorters['DIST_TO_53BP1'][1]
			sorted_df.loc[sorted_df['particle'] == particle, 'sort_flag_53bp1'] = bool

	return sorted_df
