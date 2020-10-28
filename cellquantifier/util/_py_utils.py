
def flatten(arr):

	"""

	Searches recursively for objects in an arbitarily nested array and
	returns the flattened array

	Pseudo code
	----------

	Parameters
	----------
	arr : nested list
		the nested list to recursively search for non-list objects

	Example
	----------
	"""

	def recurse(obj, out):

		if not isinstance(obj, list):
			out.append(obj)
		else:
			for x in obj:
				recurse(x, out)

		return out

	try:
		arr = list(arr)
	except TypeError:
		print('The input object was not of type list')

	flattened_arr = recurse(arr, out=[])

	return flattened_arr
