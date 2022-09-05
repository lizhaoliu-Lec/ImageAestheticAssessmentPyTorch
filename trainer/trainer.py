import torch
import torchvision
from torch.utils.data import DataLoader

from typing import List, Dict
from tqdm import tqdm
import random

from common import PROJECT_NAME
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
from dataloader import DataloaderFactory


def get_transforms(transform_settings: List[Dict]):
    transforms = []
    for transform in transform_settings:
        transforms.append(TransformFactory.instantiate(transform))
    return torchvision.transforms.Compose(transforms)


def get_train_test_dataset(dataset_setting: dict, train_transforms, test_transforms):
    dataset_setting['train_dataset']['params']['split'] = 'train'
    dataset_setting['train_dataset']['params']['transforms'] = train_transforms
    train_dataset = DatasetFactory.instantiate(dataset_setting['train_dataset'])
    dataset_setting['test_dataset']['params']['split'] = 'test'
    dataset_setting['test_dataset']['params']['transforms'] = test_transforms
    test_dataset = DatasetFactory.instantiate(dataset_setting['test_dataset'])
    for dataset in ['train_dataset', 'test_dataset']:
        dataset_setting[dataset]['params'].pop('split')
        dataset_setting[dataset]['params'].pop('transforms')
    return train_dataset, test_dataset


@TrainerFactory.register('ClassificationTrainer')
class ClassificationTrainer:
    def __init__(self, config: ConfigParser,
                 run_dir, run_id,
                 epoch, log_every,
                 gpu, seed,
                 base_lr=None, head_lr=None):
        self.config = config
        self.run_dir = run_dir
        self.run_id = run_id
        self.epoch = epoch
        self.log_every = log_every
        self.gpu = gpu
        self.seed = seed
        self.base_lr = base_lr
        self.head_lr = head_lr

        # fix the seed for reproduction
        self.set_seed()

        # get the train transforms
        train_transforms = get_transforms(self.config.train_transforms)
        test_transforms = get_transforms(self.config.test_transforms)

        # get the train/test dataset
        train_dataset, test_dataset = get_train_test_dataset(
            {'train_dataset': self.config.train_dataset, 'test_dataset': self.config.test_dataset},
            train_transforms=train_transforms,
            test_transforms=test_transforms)
        # logger=logging.getLogger()
        logger = logging.getLogger(PROJECT_NAME)
        logger.info("Training dataset: \n%s" % train_dataset)
        logger.info("Test dataset: \n%s" % test_dataset)

        # get the train/test loader
        self.config.train_dataloader['params']['dataset'] = train_dataset
        self.train_loader = DataloaderFactory.instantiate(self.config.train_dataloader)
        self.config.test_dataloader['params']['dataset'] = test_dataset
        self.test_loader = DataloaderFactory.instantiate(self.config.test_dataloader)

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
        logger.info("Using optimizer: %s" % self.config.get_name_with_params(self.config.optimizer))
        if self.base_lr is not None and self.head_lr is not None:
            logger.info("Using different learning rate for base (%.4f) and head (%.4f)" % (self.base_lr, self.head_lr))
            self.config.optimizer['params']['params'] = [{'params': self.model.base.parameters(), 'lr': self.base_lr},
                                                         {'params': self.model.head.parameters(), 'lr': self.head_lr}]
        else:
            self.config.optimizer['params']['params'] = self.model.parameters()
        self.optimizer = OptimizerFactory.instantiate(self.config.optimizer)
        self.config.optimizer['params'].pop('params')

        # get the scheduler
        logger.info("Using lr scheduler: %s" % self.config.get_name_with_params(self.config.lr_scheduler))
        self.config.lr_scheduler['params']['optimizer'] = self.optimizer
        self.lr_scheduler = LRSchedulerFactory.instantiate(self.config.lr_scheduler)
        self.config.lr_scheduler['params'].pop('optimizer')

        # get the loss function
        logger.info("Using loss: %s" % self.config.get_name_with_params(self.config.loss))
        self.loss = LossFactory.instantiate(self.config.loss)
        self.loss_meter = AvgMeter()

        # get the metric
        logger.info("Using metric: %s" % self.config.get_name_with_params(self.config.metric))
        self.metric = MetricFactory.instantiate(self.config.metric)
        self.metric_meter = AvgMeter()

        # get the tensorboard
        logger.info("Saving tensorboard file to path: %s" % self.config.save_dir)
        self.tensorboard = Tensorboard(logdir=self.config.save_dir)

        self.global_step = 0
        self.best_metric = -1

    def train(self):
        logger = logging.getLogger(PROJECT_NAME)
        logger.info("Start training...")
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

            if self.global_step % self.log_every == 0:
                # log to tensorboard
                self.tensorboard.add_scalar('Train/Loss-Step', self.loss_meter.cur, global_step=self.global_step)
                self.tensorboard.add_scalar('Train/Metric-Step', self.metric_meter.cur, global_step=self.global_step)
                if self.optimizer.__class__.__name__ != 'AdamWarmup':
                    for lr_id, lr in enumerate(self.lr_scheduler.get_lr()):
                        self.tensorboard.add_scalar('Train/LR%d' % lr_id, lr, global_step=self.global_step)
                else:
                    for lr_id, lr in enumerate(self.optimizer.get_lr()):
                        self.tensorboard.add_scalar('Train/LR%d' % lr_id, lr, global_step=self.global_step)
            self.global_step += 1

    def train_one_epoch(self, epoch):
        logger = logging.getLogger(PROJECT_NAME)
        logger.info("Start training for epoch: %d" % epoch)

        self.optimizer.step()  # In PyTorch 1.1.0 and later,
        self.lr_scheduler.step(epoch=epoch)
        for x, target in tqdm(self.train_loader):
            self.step(x, target, train=True)

        self.tensorboard.add_scalar('Train/Loss-Epoch', self.loss_meter.avg, global_step=epoch)
        self.tensorboard.add_scalar('Train/Metric-Epoch', self.metric_meter.avg, global_step=epoch)
        logger.info("Train (Epoch %d): [Loss-Epoch: %.4f] | [Metric-Epoch: %.4f]" % (epoch,
                                                                                     self.loss_meter.avg,
                                                                                     self.metric_meter.avg))
        self.loss_meter.reset()
        self.metric_meter.reset()

        logger.info("End training for epoch: %d" % epoch)

    @torch.no_grad()
    def test(self, epoch):
        logger = logging.getLogger(PROJECT_NAME)
        logger.info("Start test for epoch: %d" % epoch)

        for x, target in tqdm(self.test_loader):
            self.step(x, target, train=False)

        self.tensorboard.add_scalar('Test/Loss-Epoch', self.loss_meter.avg, global_step=epoch)
        self.tensorboard.add_scalar('Test/Metric-Epoch', self.metric_meter.avg, global_step=epoch)
        logger.info("Test (Epoch %d): [Loss-Epoch: %.4f] | [Metric-Epoch: %.4f]" % (epoch,
                                                                                    self.loss_meter.avg,
                                                                                    self.metric_meter.avg))

        self.save_checkpoint(epoch, self.metric_meter.avg, self.metric_meter.avg > self.best_metric)

        if self.metric_meter.avg > self.best_metric:
            self.best_metric = self.metric_meter.avg

        self.loss_meter.reset()
        self.metric_meter.reset()

        logger.info("End test for epoch: %d" % epoch)

    def save_checkpoint(self, epoch, metric, is_best):
        logger = logging.getLogger(PROJECT_NAME)

        save_dir = join(self.config.save_dir, 'best_model.bin' if is_best else 'latest_model.bin')
        if is_best:
            logger.info("** Saving the best model to %s with metric: %.4f" % (save_dir, metric))
        torch.save({
            'epoch': epoch,
            'metric': metric,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
        }, save_dir)

    def set_seed(self):
        seed = self.seed
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@TransformFactory.register('MultiPatchTrainer')
class MultiPatchTrainer(ClassificationTrainer):
    def __init__(self):
        super().__init__()

    def step(self, x, target, train=True):

        bs = x.size(0)
        if self.gpu:
            x = x.cuda()

        prediction = self.model(x)

        if train:
            if self.gpu:
                target = [_.cuda() for _ in target]

            loss = self.loss(prediction, target)
            metric = self.metric(prediction, target[-1])
            self.loss_meter.add(loss.item(), bs)
            self.metric_meter.add(metric, bs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.global_step % self.log_every == 0:
                # log to tensorboard
                self.tensorboard.add_scalar('Train/Loss-Step', self.loss_meter.cur, global_step=self.global_step)
                self.tensorboard.add_scalar('Train/Metric-Step', self.metric_meter.cur, global_step=self.global_step)
                if self.optimizer.__class__.__name__ != 'AdamWarmup':
                    for lr_id, lr in enumerate(self.lr_scheduler.get_lr()):
                        self.tensorboard.add_scalar('Train/LR%d' % lr_id, lr, global_step=self.global_step)
                else:
                    for lr_id, lr in enumerate(self.optimizer.get_lr()):
                        self.tensorboard.add_scalar('Train/LR%d' % lr_id, lr, global_step=self.global_step)
            self.global_step += 1
        else:
            if self.gpu:
                target = target.cuda()
            loss = self.loss(prediction, target)
            metric = self.metric(prediction, target)
            self.loss_meter.add(loss.item(), bs)
            self.metric_meter.add(metric, bs)
