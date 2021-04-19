import os

join = os.path.join


def check_if_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File `%s` not found" % file_path)


def check_if_has_required_args(_dict: dict, keys: list):
    for k in keys:
        if k not in _dict:
            raise ValueError('Required arguments `%s` not found.' % k)


def mkdirs_if_not_exist(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=False)


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton
