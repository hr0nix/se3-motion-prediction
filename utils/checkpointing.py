import os

import torch

import logging
logger = logging.getLogger(__name__)


def try_load_checkpoint(model, optimizer, checkpoint_file):
    if not os.path.exists(checkpoint_file):
        logger.info(f'No checkpoint file found at {checkpoint_file}.')
        return 0

    logger.info(f'Loading checkpoint file {checkpoint_file}.')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def save_checkpoint(model, optimizer, checkpoint_file, epoch):
    logger.info(f'Saving checkpoint file {checkpoint_file}.')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_file)
