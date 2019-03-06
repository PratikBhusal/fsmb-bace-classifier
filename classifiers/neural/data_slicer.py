
def slice_data(data, slice_length=40, overlap_percent=None):
	"""

	:param data: a list of tokens
	:param slice_length: how long each slice should be
	:param overlap_percent: what percent of the slices will be overlap with the adjacent slices
	:return: [slice1, slice2, ...] where slicei is [token1, token2, token3...]
	"""
	if overlap_percent is not None and (overlap_percent >= 1 or overlap_percent <= 0):
		raise Exception("Invalid overlap amount")

	step = slice_length if overlap_percent is None else int(slice_length * (1 - overlap_percent))
	slices = [data[i:min(len(data), i+slice_length)] for i in range(0, len(data), step)]
	return slices
