import os
import torch
import logging
from pathlib import Path
import torch.distributed as dist
from pytorch_lightning import LightningModule
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class Average:
    def __init__(self, value=0.0, weight=0):
        self.sum = weight * value
        self.wgt = weight

    def update(self, value, weight):
        self.sum += weight * value
        self.wgt += weight

    def __call__(self):
        return 0 if self.wgt <= 0 else self.sum / self.wgt

    def __repr__(self):
        return f'sum: {self.sum}, wgt: {self.wgt}, avr: {self()}'

    def __str__(self):
        return self.__repr__()


class Statistics:
    def __init__(self):
        self.data = dict()
        self.weight = 0

    def clear(self):
        self.data = dict()
        self.weight = 0

    def update(self, data, weight):
        if isinstance(weight, torch.Tensor):
            weight = weight.item()
        for key, value in data.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if key in self.data:
                    self.data[key].update(value, weight)
                else:
                    self.data[key] = Average(value, weight)
        self.weight += weight

    def __call__(self):
        data = dict()
        for key, aver in self.data.items():
            data[key] = aver()
        return data


class BaseModule(LightningModule):
    def __init__(
            self, valid_dataset, train_dataset,
            num_workers, pin_memory,
            valid_metric, train_metric,
            valid_bar_keys, valid_log_keys,
            train_bar_keys, train_log_keys, log_aver,
    ):
        super().__init__()
        logging.info('Initialising base Lightning module')
        self.valid_dataset = valid_dataset
        self.train_dataset = train_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.valid_metric = valid_metric
        self.train_metric = train_metric
        self.valid_bar_keys = valid_bar_keys
        self.valid_log_keys = valid_log_keys
        self.train_bar_keys = train_bar_keys
        self.train_log_keys = train_log_keys
        self.log_aver = log_aver
        self.valid_stats = Statistics()
        self.train_stats = Statistics()
        self.metrics_writer = None
        logging.info('Base Lightning module initialization done')

    @property
    def in_ddp(self):
        try:
            logging.debug('Trying to get world size')
            size = dist.get_world_size()
            logging.info(f'World size is {size}')
            logging.info('Enabling distributed mode')
            return True
        except RuntimeError:
            logging.debug('Failed to get world size')
            logging.info('Disabling distributed mode')
            return False

    def setup(self, stage: str) -> None:
        if self.valid_metric is not None:
            self.valid_metric = self.valid_metric.to(device=self.device)
        if self.train_metric is not None:
            self.train_metric = self.train_metric.to(device=self.device)

    def create_writer(self):
        if self.metrics_writer is None:
            train_dir = Path(self.trainer.default_root_dir)
            metrics_dir = train_dir / 'metrics'
            if not metrics_dir.exists():
                logging.debug(f'Creating directory {metrics_dir}')
                metrics_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_writer = SummaryWriter(log_dir=str(metrics_dir), comment=str(train_dir.name))

    def close_writer(self):
        if self.metrics_writer is not None:
            self.metrics_writer.close()
            self.metrics_writer = None

    def report_scalar(self, title, tag, iteration, value):
        if self.metrics_writer is None:
            self.create_writer()
        self.metrics_writer.add_scalars(main_tag=title, tag_scalar_dict={tag: value}, global_step=iteration)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        logging.info('Creating train DataLoader')
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=None,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        return loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        logging.info('Creating validation DataLoader')
        loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        return loader

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            average = self.valid_stats()
            self.valid_stats.clear()
            log_dict = dict()
            for key, key_log in self.valid_bar_keys.items():
                if key in average:
                    value = average[key]
                    log_dict[key_log] = value
                    self.report_scalar(title=key, tag='valid', iteration=self.global_step, value=value)
            self.log_dict(log_dict, prog_bar=True)
            log_dict = dict()
            for key, key_log in self.valid_log_keys.items():
                if key in average:
                    value = average[key]
                    log_dict[key_log] = value
                    self.report_scalar(title=key, tag='valid', iteration=self.global_step, value=value)
            self.log_dict(log_dict)

    def on_train_epoch_end(self) -> None:
        ckpt_name = f'epoch={self.trainer.current_epoch}-step={self.trainer.global_step}.ckpt'
        ckpt_path = os.path.join(self.trainer.default_root_dir, 'checkpoints', ckpt_name)
        if not os.path.exists(ckpt_path):
            self.trainer.save_checkpoint(ckpt_path)

    def on_fit_end(self) -> None:
        self.close_writer()
