from ...util import flatten

def add_stat_anno(ax, x_arr,
				  y_arr,
				  p_val_arr,
				  bwidth=.1,
				  bheight=.1,
				  delta=.05,
				  fontsize=10,
				  star_spacing=.05,
				  show_brackets=True):

	"""
	Add statistical annotations to the axis i.e. p-value stars and brackets

	Pseudo code
	----------
	1. Flatten all input arrays
	2. Convert p-values to strings of stars and add to axis

	Parameters
	----------
	ax : object
		matplotlib axis.

	x_arr : list
		list of x-coordinates for the stars and/or brackets in axis units

	y_arr : list
		a list of y-coordiantes for the stars and/or brackets in axis units

	p_val_arr : list
		a list of the pvalues e.g. .001, .00001, etc.

	bwidth : float, optional
		width of the bracket in axis units

	bheight : float, optional
		height of a bracket in axis units

	delta : float, optional
		padding between y-coordinate and where the bracket is placed

	fontsize : float, optional
		fontsize for the stars

	star_spacing : float, optional
		spacing between star and bracket

	show_brackets : bool, optional
		whether or not to show the bracket


	Example
	----------
	"""

	# """
	# ~~~~~~~~~~~Flatten all input arrays~~~~~~~~~~~~~~
	# """

	x_arr = flatten(x_arr)
	y_arr = flatten(y_arr)
	p_val_arr = flatten(p_val_arr)

	# """
	# ~~~~~~~~~~~Convert p-value to star format~~~~~~~~~~~~~~
	# """

	def pval_to_stars(pval):

		if pval <= .0001:
			t_test_str = '****'
			shift = .075
		elif pval > .0001 and pval <= .001:
			t_test_str = '***'
			shift = .05
		elif pval > .001 and pval <= .01:
			t_test_str = '**'
			shift = .025
		elif pval > .01 and pval <= .05:
			t_test_str = '*'
			shift = .01
		elif pval > .05:
			t_test_str = 'ns'
			shift = .025

		return t_test_str, shift

	# """
	# ~~~~~~~~~~~Add stars to axis, iteratively~~~~~~~~~~~~~~
	# """

	for x,y,p in zip(x_arr,y_arr,p_val_arr):

		bracket_x = [x-bwidth, x-bwidth, x+bwidth, x+bwidth]
		bracket_y = [y+delta, y+bheight+delta, y+bheight+delta, y+delta]

		t_test_str, shift = pval_to_stars(p)

		if show_brackets:
			ax.plot(bracket_x, bracket_y, c='black')
			bracket_y = [y+delta, y+bheight+delta, y+bheight+delta, y+delta]
			ax.text(x-shift, y+bheight+delta+star_spacing, t_test_str, fontsize=fontsize)
		else:
			ax.text(x-shift, y+delta+star_spacing, t_test_str, fontsize=fontsize)
