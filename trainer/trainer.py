import torch
import torchvision
from torch.utils.data import DataLoader

from typing import List, Dict
from tqdm import tqdm

from common.avg_meter import AvgMeter
from common.logging import logging
from common.tensorboard import Tensorboard
from common.utils import join

from trainer import TrainerFactory
from trainer.config_parser import ConfigParser
from trainer.utils import DataParallel, get_gpu_device

from transform import TransformFactory
from dataset import DatasetFactory
from model import ModelFactory
from optimizer import OptimizerFactory
from lr_scheduler import LRSchedulerFactory
from loss import LossFactory
from metric import MetricFactory


def get_transforms(transform_settings: List[Dict]):
    transforms = []
    for transform in transform_settings:
        transforms.append(TransformFactory.instantiate(transform))
    return torchvision.transforms.Compose(transforms)


def get_train_test_dataset(dataset_setting: Dict, train_transforms, test_transforms):
    dataset_setting['params']['split'] = 'train'
    dataset_setting['params']['transforms'] = train_transforms
    train_dataset = DatasetFactory.instantiate(dataset_setting)
    dataset_setting['params']['split'] = 'test'
    dataset_setting['params']['transforms'] = test_transforms
    test_dataset = DatasetFactory.instantiate(dataset_setting)
    dataset_setting['params'].pop('split')
    dataset_setting['params'].pop('transforms')
    return train_dataset, test_dataset


@TrainerFactory.register('ClassificationTrainer')
class ClassificationTrainer:
    def __init__(self, config: ConfigParser, run_dir, run_id, batch_size, epoch, gpu, log_every, num_workers):
        self.config = config
        self.run_dir = run_dir
        self.run_id = run_id
        self.batch_size = batch_size
        self.epoch = epoch
        self.gpu = gpu
        self.log_every = log_every
        self.num_workers = num_workers

        # get the train transforms
        train_transforms = get_transforms(self.config.train_transforms)
        test_transforms = get_transforms(self.config.test_transforms)

        # get the train/test dataset
        train_dataset, test_dataset = get_train_test_dataset(self.config.dataset,
                                                             train_transforms=train_transforms,
                                                             test_transforms=test_transforms)
        logging.info("Training dataset: \n%s" % train_dataset)
        logging.info("Test dataset: \n%s" % test_dataset)

        # get the train/test loader
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                       num_workers=self.num_workers,
                                       shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=False, pin_memory=True)

        # get the model
        self.model = ModelFactory.instantiate(self.config.model)
        self.num_gpu = len(self.gpu)
        if self.num_gpu > 0:
            if self.num_gpu == 1:
                self.model.cuda(get_gpu_device(self.gpu[0]))
            else:
                self.model = DataParallel(module=self.model, device_ids=[get_gpu_device(_) for _ in self.gpu])
                self.model.cuda(get_gpu_device(self.gpu[0]))

        # get the optimizer
        logging.info("Using optimizer: %s" % self.config.get_name_with_params(self.config.optimizer))
        self.config.optimizer['params']['params'] = self.model.parameters()
        self.optimizer = OptimizerFactory.instantiate(self.config.optimizer)
        self.config.optimizer['params'].pop('params')

        # get the scheduler
        logging.info("Using lr scheduler: %s" % self.config.get_name_with_params(self.config.lr_scheduler))
        self.config.lr_scheduler['params']['optimizer'] = self.optimizer
        self.lr_scheduler = LRSchedulerFactory.instantiate(self.config.lr_scheduler)
        self.config.lr_scheduler['params'].pop('optimizer')

        # get the loss function
        logging.info("Using loss: %s" % self.config.get_name_with_params(self.config.loss))
        self.loss = LossFactory.instantiate(self.config.loss)
        self.loss_meter = AvgMeter()

        # get the metric
        logging.info("Using metric: %s" % self.config.get_name_with_params(self.config.metric))
        self.metric = MetricFactory.instantiate(self.config.metric)
        self.metric_meter = AvgMeter()

        # get the tensorboard
        logging.info("Saving tensorboard file to path: %s" % self.config.save_dir)
        self.tensorboard = Tensorboard(logdir=self.config.save_dir)

        self.global_step = 0
        self.best_metric = -1

    def train(self):
        logging.info("Start training...")
        for epoch in range(1, self.epoch + 1):
            self.train_one_epoch(epoch)

            self.test(epoch)

    def step(self, x, target, train=True):

        bs = x.size(0)
        if self.gpu:
            x = x.cuda()

        prediction = self.model(x)

        if self.gpu:
            target = target.cuda()

        loss = self.loss(prediction, target)
        metric = self.metric(prediction, target)
        self.loss_meter.add(loss.item(), bs)
        self.metric_meter.add(metric, bs)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log to tensorboard
            self.tensorboard.add_scalar('Train/Loss-Step', self.loss_meter.cur, global_step=self.global_step)
            self.tensorboard.add_scalar('Train/Metric-Step', self.metric_meter.cur, global_step=self.global_step)
            for lr in self.lr_scheduler.get_lr():
                self.tensorboard.add_scalar('Train/LR', lr, global_step=self.global_step)

            self.global_step += 1

    def train_one_epoch(self, epoch):
        logging.info("Start training for epoch: %d" % epoch)

        self.optimizer.step()  # In PyTorch 1.1.0 and later,
        self.lr_scheduler.step(epoch=epoch)
        for x, target in tqdm(self.train_loader):
            self.step(x, target, train=True)

        self.tensorboard.add_scalar('Train/Loss-Epoch', self.loss_meter.avg, global_step=epoch)
        self.tensorboard.add_scalar('Train/Metric-Epoch', self.metric_meter.avg, global_step=epoch)
        logging.info("Train (Epoch %d): [Loss-Epoch: %.4f] | [Metric-Epoch: %.4f]" % (epoch,
                                                                                      self.loss_meter.avg,
                                                                                      self.metric_meter.avg))
        self.loss_meter.reset()
        self.metric_meter.reset()

        logging.info("End training for epoch: %d" % epoch)

    @torch.no_grad()
    def test(self, epoch):
        logging.info("Start test for epoch: %d" % epoch)

        for x, target in tqdm(self.test_loader):
            self.step(x, target, train=False)

        self.tensorboard.add_scalar('Test/Loss-Epoch', self.loss_meter.avg, global_step=epoch)
        self.tensorboard.add_scalar('Test/Metric-Epoch', self.metric_meter.avg, global_step=epoch)
        logging.info("Test (Epoch %d): [Loss-Epoch: %.4f] | [Metric-Epoch: %.4f]" % (epoch,
                                                                                     self.loss_meter.avg,
                                                                                     self.metric_meter.avg))

        self.save_checkpoint(epoch, self.metric_meter.avg, self.metric_meter.avg > self.best_metric)

        if self.metric_meter.avg > self.best_metric:
            self.best_metric = self.metric_meter.avg

        self.loss_meter.reset()
        self.metric_meter.reset()

        logging.info("End test for epoch: %d" % epoch)

    def save_checkpoint(self, epoch, metric, is_best):
        save_dir = join(self.config.save_dir, 'best_model.bin' if is_best else 'latest_model.bin')
        if is_best:
            logging.info("** Saving the best model to %s with metric: %.4f" % (save_dir, metric))
        torch.save({
            'epoch': epoch,
            'metric': metric,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
        }, save_dir)
