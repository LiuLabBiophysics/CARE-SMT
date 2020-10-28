import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_heat_map(ax,
				 r_discrete,
				 r_cont,
				 f_cont,
				 ylabel=None,
				 nbins=8,
				 hole_size=10,
				 edge_ring=False,
				 pixel_size=.1083,
				 range=None,
				 show_colorbar=True):


	"""Add heat maps to the axis

	Parameters
	----------
	ax: object
		matplotlib axis to annotate

	r_discrete: 1d ndarray
		discrete domain before interpolation

	r_cont: str
		'continuous' domain after interpolation

	f_cont: str
		'continuous' function after interpolation, f_cont = F(r_cont)

	"""

	ring_radius = np.abs(r_cont.min())

	# """
	# ~~~~~~~~~~~Generate cartesian heat map~~~~~~~~~~~~~~
	# """

	ntheta = 100
	theta = np.linspace(0,2*np.pi,ntheta)
	r_cont, theta = np.meshgrid(r_cont, theta)
	_r = np.tile(f_cont, (ntheta, 1))
	_r[_r == 0] = None

	local_range = (f_cont[f_cont != 0].min(), f_cont[f_cont != 0].max())

	# """
	# ~~~~~~~~~~~Plot the heat map as polar~~~~~~~~~~~~~~
	# """

	ax.set_yticks(r_discrete)
	ax.set_xticklabels([])
	ax.set_yticklabels([])

	if range:
		cb = ax.pcolormesh(theta, r_cont, _r, \
						   cmap='coolwarm',
						   vmin=range[0],
						   vmax=range[1])
	else:
		cb = ax.pcolormesh(theta, r_cont, _r, \
						   cmap='coolwarm',
						   vmin=local_range[0],
						   vmax=local_range[1])

	# """
	# ~~~~~~~~~~~Color Bar~~~~~~~~~~~~~~
	# """

	if show_colorbar:
		cb = plt.colorbar(cb, ax=ax, extend='both')
		cb.outline.set_visible(False)
		cb.set_label(ylabel)
		
	ax.grid(True, axis='y', color='black', linewidth=.5)

	# """
	# ~~~~~~~~~~~Hole~~~~~~~~~~~~~~
	# """

	hole = plt.Circle((0, 0), radius=hole_size, \
						 transform=ax.transData._b, color='black')
	ax.add_artist(hole)

	# """
	# ~~~~~~~~~~~Edge Ring~~~~~~~~~~~~~~
	# """

	if edge_ring:
		edge = plt.Circle((0, 0), radius=hole_size+ring_radius,\
						  transform=ax.transData._b, \
					      color='yellow', fill=False, linewidth=1)
		ax.add_artist(edge)

	# """
	# ~~~~~~~~~~~Bin Size Text~~~~~~~~~~~~~
	# """

	bin_size = round(pixel_size*(r_discrete[1]-r_discrete[0]), 2)
	bin_sz_str = r'$\mathbf{Bin Size: %s\mu m}$' % str(bin_size)

	# ax.text(.2,
	# 	    1,
	# 	    bin_sz_str,
	# 	    horizontalalignment='right',
	# 	    verticalalignment='bottom',
	# 	    fontsize = 10,
	# 	    color = (0, 0, 0, 1),
	# 	    transform=ax.transAxes)
