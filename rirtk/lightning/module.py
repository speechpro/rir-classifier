import logging
from .base import BaseModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT


class Module(BaseModule):
    def __init__(
        self, model,
        valid_dataset=None, train_dataset=None,
        num_workers=0, pin_memory=False,
        optimizer=None, scheduler=None, criterion=None,
        valid_metric=None, train_metric=None,
        valid_bar_keys=None, valid_log_keys=None,
        train_bar_keys=None, train_log_keys=None, log_aver=256
    ):
        super().__init__(
            valid_dataset=valid_dataset, train_dataset=train_dataset,
            num_workers=num_workers, pin_memory=pin_memory,
            valid_metric=valid_metric, train_metric=train_metric,
            valid_bar_keys=valid_bar_keys, valid_log_keys=valid_log_keys,
            train_bar_keys=train_bar_keys, train_log_keys=train_log_keys, log_aver=log_aver,
        )
        logging.info('Initialising Lightning module')
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        logging.info('Lightning module initialization done')

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        inputs = batch['inputs']
        labels = batch['labels']
        outputs, outputs_len = self.model(inputs, None)
        loss = self.criterion(outputs, labels)
        weight = labels.shape[0]
        stats = {
            'loss': loss.item(),
            'lr': self.scheduler.get_last_lr()[0],
        }
        if self.train_metric is not None:
            stats[self.train_metric.label] = self.train_metric(outputs, labels)
        self.train_stats.update(stats, weight)
        if self.train_stats.weight >= self.log_aver:
            average = self.train_stats()
            self.train_stats.clear()
            log_dict = dict()
            for key, key_log in self.train_bar_keys.items():
                if key in average:
                    value = average[key]
                    log_dict[key_log] = value
                    self.report_scalar(title=key, tag='train', iteration=self.global_step, value=value)
            self.log_dict(log_dict, prog_bar=True)
            log_dict = dict()
            for key, key_log in self.train_log_keys.items():
                if key in average:
                    value = average[key]
                    log_dict[key_log] = value
                    self.report_scalar(title=key, tag='train', iteration=self.global_step, value=value)
            self.log_dict(log_dict)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        inputs = batch['inputs']
        labels = batch['labels']
        outputs, outputs_len = self.model(inputs, None)
        loss = self.criterion(outputs, labels)
        if not self.trainer.sanity_checking:
            weight = labels.shape[0]
            stats = {'loss': loss.item()}
            if self.valid_metric is not None:
                stats[self.valid_metric.label] = self.valid_metric(outputs, labels)
            self.valid_stats.update(stats, weight)
        return loss
