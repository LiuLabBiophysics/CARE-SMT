from scipy.stats import ttest_ind

def t_test(a,b):

	"""Performed a t-test on two samples

	Parameters
	----------
	a,b: arrays of values to be used for the t-test

	Returns
	-------
	t: t-value
    p: two-tailed p value

	Examples
	--------
	>>>from cellquantifier.plot import plot_d_hist
	>>>path1 = 'cellquantifier/data/test_d_values1.csv'
	>>>path2 = 'cellquantifier/data/test_d_values2.csv'
	>>>plot_d_hist(damaged=path1, control=path2)

	"""
	t = ttest_ind(a, b)

	return t
