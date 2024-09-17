import os
import re
import glob
import logging
from pathlib import Path
from pytorch_lightning import Trainer as LightningTrainer


def train(module, num_epochs, stop_after=0, resume_training=True, cml_task=None, **kwargs):
    assert 'default_root_dir' in kwargs, f'The default_root_dir parameter is not specified in the config\n{kwargs}'
    work_dir = kwargs['default_root_dir']
    ckpt_path = None
    kwargs['max_epochs'] = num_epochs
    defaults = {
        'reload_dataloaders_every_n_epochs': 1,
    }
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    if resume_training:
        path_mask = os.path.join(work_dir, 'checkpoints/epoch=*-step=*.ckpt')
        pattern = re.compile(r'^.*epoch=(\d+)-step=(\d+).ckpt$')
        ckpt_paths = list()
        for path in glob.iglob(path_mask, recursive=True):
            match = pattern.match(path)
            if match is not None:
                epoch = int(match.group(1))
                step = int(match.group(2))
                ckpt_paths.append((epoch, step, path))
        if len(ckpt_paths) == 0:
            logging.info(f'Failed to find any checkpoints in {work_dir}')
            if stop_after > 0:
                kwargs['max_epochs'] = min(stop_after, num_epochs)
        else:
            ckpt_paths = list(sorted(ckpt_paths, reverse=True))
            epoch, _, ckpt_path = ckpt_paths[0]
            if stop_after > 0:
                kwargs['max_epochs'] = min(epoch + 1 + stop_after, num_epochs)
            logging.info(f'Starting training from checkpoint {ckpt_path}')
    trainer = LightningTrainer(**kwargs)
    trainer.fit(model=module, ckpt_path=ckpt_path)
    if module.current_epoch == num_epochs:
        path = Path(work_dir) / '.done'
        path.touch()
    if cml_task is not None:
        if module.current_epoch < num_epochs:
            cml_task.mark_stopped(force=True)
        else:
            cml_task.close()
