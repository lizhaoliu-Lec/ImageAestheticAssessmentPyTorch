import argparse

from trainer import TrainerFactory
from trainer.config_parser import ConfigParser

def arg_parser():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--config',
                        type=str,
                        default='config/classification.yaml',
                        help='Configuration file to use.')

    return parser.parse_args()


if __name__ == '__main__':
    arg = arg_parser()

    config = ConfigParser(config_path=arg.config)
    config.trainer['params']['config'] = config
    trainer = TrainerFactory.instantiate(config.trainer)

    trainer.train()
