class FactoryBase:

    def __init__(self):
        self._dict = {}

    def __getitem__(self, key):
        if key not in self._dict:
            raise ValueError("Unsupported %s" % key)
        return self._dict[key]

    def __setitem__(self, key, value):
        if key not in self._dict:
            self._dict[key] = value

    def register(self, key):
        def wrapper(dataset_class):
            self[key] = dataset_class
            return dataset_class

        return wrapper

    def _instantiate(self, key, **kwargs):
        return self[key](**kwargs)

    def instantiate(self, config):
        return self[config['name']](**config['params'])

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()
