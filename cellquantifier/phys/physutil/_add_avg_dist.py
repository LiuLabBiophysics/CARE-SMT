import pandas as pd

def add_avg_dist(df):

	"""
	Add avg_dist_53bp1 and avg_dist_bound columns to DataFrame

	Parameters
	----------
	df : DataFrame
		DataFrame with column 'particle', 'dist_to_boundary', 'dist_to_53bp1'

	"""

	particles = df['particle'].unique()
	avg_dist_53bp1 = df.groupby('particle')['dist_to_53bp1'].mean()
	avg_dist_boundary = df.groupby('particle')['dist_to_boundary'].mean()

	for particle in particles:
		dist_53bp1 = avg_dist_53bp1[particle]
		dist_bound = avg_dist_boundary[particle]
		df.loc[df['particle'] == particle, 'avg_dist_53bp1'] = dist_53bp1
		df.loc[df['particle'] == particle, 'avg_dist_bound'] = dist_bound

	return df
