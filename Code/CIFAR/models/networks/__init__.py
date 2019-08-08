from models.networks.PyramidTreeNet import PyramidTreeNet


def get_net_by_name(name):
	if name == PyramidTreeNet.__name__:
		return PyramidTreeNet
	else:
		raise ValueError('unrecognized type of network: %s' % name)
