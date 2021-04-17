import os

join = os.path.join


def check_if_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File `%s` not found" % file_path)
