import os

from tfrecord.torch.dataset import MultiTFRecordDataset
from torch.utils.data import DataLoader

from .transforms import make_transform_chain


def create_data_loader(path, batch_size, device, shuffle_queue_size=None):
    dataset_files = os.listdir(path)
    splits = {
        file: 1.0 for file in dataset_files
    }
    dataset = MultiTFRecordDataset(
        data_pattern=path + '/{}',
        index_pattern=None,
        splits=splits,
        infinite=False,
        shuffle_queue_size=shuffle_queue_size,
        transform=make_transform_chain(device),
    )
    return DataLoader(dataset, batch_size=batch_size)
