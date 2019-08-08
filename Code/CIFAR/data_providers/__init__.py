from data_providers.cifar import Cifar10DataProvider, Cifar100DataProvider


def get_data_provider_by_name(name, train_params):
	""" Return required data provider class """
	if name == Cifar10DataProvider.name():
		return Cifar10DataProvider(**train_params)
	elif name == Cifar100DataProvider.name():
		return Cifar100DataProvider(**train_params)
	else:
		print('Sorry, data provider for `%s` dataset '
		      'was not implemented yet' % name)
		exit()
