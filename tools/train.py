import pyrallis
import torch
import tqdm

from dataclasses import dataclass

from models.se3_transformer import SE3TransformerModel
from models.losses import motion_prediction_loss
from utils.checkpointing import try_load_checkpoint, save_checkpoint
from data import create_data_loader

import logging
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    num_layers: int = 2
    max_degree: int = 3
    feature_dim: int = 32
    num_heads: int = 4
    num_modes: int = 5


@dataclass
class TrainConfig:
    train_dataset_path: str
    validation_dataset_path: str
    batch_size: int = 32
    shuffle_queue_size: int = 2048
    num_epochs: int = 100
    lr: float = 0.001
    device_name: str = 'cpu'
    checkpoint_file: str = None
    output_file: str = None
    model_config: ModelConfig = pyrallis.field(default_factory=ModelConfig)

    @property
    def device(self):
        return torch.device(self.device_name)


def create_optimizer(model, config):
    return torch.optim.Adam(model.parameters(), lr=config.lr)


def create_model(model_config):
    return SE3TransformerModel(
        num_layers=model_config.num_layers,
        max_degree=model_config.max_degree,
        feature_dim=model_config.feature_dim,
        num_heads=model_config.num_heads,
        num_modes=model_config.num_modes,
    )


@pyrallis.wrap()
def main(config: TrainConfig):
    model = create_model(config.model_config)
    optimizer = create_optimizer(model, config)

    start_epoch = 0
    if config.checkpoint_file is not None:
        start_epoch = try_load_checkpoint(model, optimizer, config.checkpoint_file)

    model.train()
    model.to(config.device)

    train_loader = create_data_loader(
        config.train_dataset_path, config.batch_size, config.device, config.shuffle_queue_size)
    validation_loader = create_data_loader(config.validation_dataset_path, config.batch_size, config.device)

    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f'Epoch {epoch + 1} started.')

        train_loss_sum = 0.0
        num_train_batches = 0
        first_train_batch = None
        for train_batch in tqdm.tqdm(train_loader):
            # TODO: remove me
            if first_train_batch is None:
                first_train_batch = train_batch
            optimizer.zero_grad()
            predictions = model(first_train_batch)
            loss = motion_prediction_loss(predictions, first_train_batch)
            train_loss_sum += loss.item()
            print(loss.item())
            num_train_batches += 1
            loss.backward()
            optimizer.step()

        logger.info(f'Train loss: {train_loss_sum / num_train_batches:0.000}')

        val_loss_sum = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for val_batch in tqdm.tqdm(validation_loader):
                predictions = model(val_batch)
                loss = motion_prediction_loss(predictions, val_batch)
                val_loss_sum += loss.item()
                num_val_batches += 1

        logger.info(f'Val loss: {val_loss_sum / num_val_batches:0.000}')

        if config.checkpoint_file is not None:
            save_checkpoint(model, optimizer, config.checkpoint_file, epoch)

    if config.output_file is not None:
        torch.save(model.state_dict(), config.output_file)


if __name__ == '__main__':
    main()
