from scipy import stats
import scipy.optimize as opt
import numpy as np
import math

def msd1(x, D, alpha):
	return 4*D*(x**alpha)

def msd2(x, D, alpha, c):
	return 4*D*(x**alpha) + c

def fit_msd1(x, y):
	popt, pcov = opt.curve_fit(msd1, x, y,
				  bounds=(0, [np.inf, np.inf]))
	return popt

def fit_msd1_log(x, y):
	x = [math.log(i) for i in x]
	y = [math.log(i) for i in y]

	slope, intercept, r, p, stderr = \
		stats.linregress(x,y)

	D = np.exp(intercept) / 4; alpha = slope
	popt = (D, alpha)

	return popt

def fit_msd2(x, y):
	popt, pcov = opt.curve_fit(msd2, x, y,
				  bounds=(0, [np.inf, np.inf, np.inf]),
				  )
	return popt
