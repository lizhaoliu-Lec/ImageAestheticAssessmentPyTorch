import os
import yaml
from typing import Dict, List

from common import PROJECT_NAME
from common.utils import check_if_file_exists, check_if_has_required_args, mkdirs_if_not_exist, join
from common.logging import set_logging
import logging
import datetime
from pprint import pformat


class ConfigParser:
    def __init__(self, config_path):
        check_if_file_exists(config_path)
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config = config
        self.now = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        check_if_has_required_args(self.config,
                                   ['logger', 'trainer', 'train_dataset', 'test_dataset', 'model',
                                    'loss', 'optimizer', 'lr_scheduler',
                                    'metric', 'train_transforms', 'train_transforms', 'train_dataloader',
                                    'test_dataloader'])

        self.fix_empty_params_of_dict(self.config['trainer'])
        self.fix_empty_params_of_dict(self.config['train_dataset'])
        self.fix_empty_params_of_dict(self.config['test_dataset'])
        self.fix_empty_params_of_dict(self.config['model'])
        self.fix_empty_params_of_dict(self.config['loss'])
        self.fix_empty_params_of_dict(self.config['optimizer'])
        self.fix_empty_params_of_dict(self.config['lr_scheduler'])
        self.fix_empty_params_of_dict(self.config['metric'])
        self.fix_empty_params_of_dict(self.config['train_dataloader'])
        self.fix_empty_params_of_dict(self.config['test_dataloader'])
        self.fix_empty_params_of_dicts(self.config['train_transforms'])
        self.fix_empty_params_of_dicts(self.config['test_transforms'])

        self.logger = self.config['logger']
        self.trainer = self.config['trainer']
        self.train_dataset = self.config['train_dataset']
        self.test_dataset = self.config['test_dataset']
        self.model = self.config['model']
        self.loss = self.config['loss']
        self.optimizer = self.config['optimizer']
        self.lr_scheduler = self.config['lr_scheduler']
        self.metric = self.config['metric']
        self.train_transforms = self.config['train_transforms']
        self.test_transforms = self.config['test_transforms']
        self.train_dataloader = self.config['train_dataloader']
        self.test_dataloader = self.config['test_dataloader']
        # self.save_dir = self.config['save_dir']
        # init the logger
        set_logging(level=self.logger['level'], name=PROJECT_NAME)

        logger = logging.getLogger(PROJECT_NAME)

        # print the config file
        logger.info("Using the following configuration: \n%s" % pformat(config))
        logger.info("Saving the training result to: %s" % self.save_dir)

        # save the config file
        self.save_config(self.save_dir, os.path.basename(config_path), config)
        logger.info("Saving the config file to: %s" % self.save_dir)

        # set gpu environment
        self.set_gpu(self.trainer)
        import torch
        logging.debug("GPU visible: %s" % str(torch.cuda.is_available()))

    @staticmethod
    def fix_empty_params_of_dict(_config: dict):
        if 'params' not in _config:
            _config['params'] = {}

    @staticmethod
    def fix_empty_params_of_dicts(_configs: List[Dict]):
        for _config in _configs:
            if 'params' not in _config:
                _config['params'] = {}

    @staticmethod
    def save_config(save_dir, config_name, config):
        with open(join(save_dir, config_name), 'w') as f:
            yaml.dump(config, f)

    @staticmethod
    def set_gpu(_trainer_config: dict):
        gpu = _trainer_config['params']['gpu']
        if isinstance(gpu, int):
            gpu = [gpu]

        gpu = [str(g) for g in gpu]

        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu)

        gpu = [str(_) for _ in range(len(gpu))]
        _trainer_config['params']['gpu'] = gpu

    @property
    def save_dir(self):
        _save_dir = join(self.trainer['params']['run_dir'],
                         self.train_dataset['name'],
                         self.trainer['params']['run_id'] + '-' + self.now)
        mkdirs_if_not_exist(_save_dir)
        return _save_dir

    @property
    def gpu(self):
        return self.trainer['params']['gpu']

    @staticmethod
    def get_name_with_params(_config):
        return _config['name'] + '(' + ', '.join(["%s=%s" % (k, v) for k, v in _config['params'].items()]) + ')'


if __name__ == '__main__':
    def run_config_parser():
        config_parser = ConfigParser(
            config_path='E:/Projects/Paper/ImageAestheticAssessmentPyTorch/config/classification.yaml')


    run_config_parser()
