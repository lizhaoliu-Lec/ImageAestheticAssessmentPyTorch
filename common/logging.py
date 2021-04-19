import logging

from common.utils import Singleton


@Singleton
def set_logging(level='INFO', name='IAAPyTorch'):
    FORMAT = '[{0} %(asctime)s %(filename)s %(lineno)d %(levelname)s] %(message)s'.format(name)
    logging.basicConfig(level=level,
                        datefmt='%m/%d %H:%M:%S',
                        format=FORMAT)
