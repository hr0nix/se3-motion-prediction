import torch


def remove_unused_features(sample):
    keys_to_leave = [
        'state/current/x',
        'state/current/y',
        'state/current/z',
        'state/current/bbox_yaw',
        'state/current/velocity_x',
        'state/current/velocity_y',
        'state/current/valid',
        'state/future/x',
        'state/future/y',
        'state/future/z',
        'state/future/valid',
        'state/tracks_to_predict',
        'state/type',
    ]

    return {
        key: sample[key]
        for key in keys_to_leave
    }


def to_device(sample, device):
    return {
        key: torch.as_tensor(value, device=device)
        for key, value in sample.items()
    }


def postprocess_features(sample):
    # TODO: it would be faster to do this after batching

    valid = sample['state/current/valid'].type(torch.bool)
    future_valid = sample['state/future/valid'].reshape(128, 80).type(torch.bool)

    coords = torch.concat(
        [
            sample['state/current/x'].reshape(128, 1),
            sample['state/current/y'].reshape(128, 1),
            sample['state/current/z'].reshape(128, 1)
        ],
        dim=-1,
    )
    velocity = torch.concat(
        [
            sample['state/current/velocity_x'].reshape(128, 1, 1),
            sample['state/current/velocity_y'].reshape(128, 1, 1),
            torch.zeros(128, 1, 1),
        ],
        dim=-1,
    )
    dir = torch.concat(
        [
            torch.cos(sample['state/current/bbox_yaw']).reshape(128, 1, 1),
            torch.sin(sample['state/current/bbox_yaw']).reshape(128, 1, 1),
            torch.zeros(128, 1, 1),
        ],
        dim=-1,
    )
    future_coords = torch.concat(
        [
            sample['state/future/x'].reshape(128, 80, 1),
            sample['state/future/y'].reshape(128, 80, 1),
            sample['state/future/z'].reshape(128, 80, 1),
        ],
        dim=-1,
    )

    return {
        'tracks_to_predict': sample['state/tracks_to_predict'],
        'valid': valid,
        'future_valid': future_valid,
        'coords': coords,
        'future_coords': future_coords,
        'type_0_categorical': sample['state/type'].type(torch.int).reshape(128, 1),
        'type_1': torch.concat([velocity, dir], dim=-2),
    }


def make_transform_chain(device):
    def chain(sample):
        sample = remove_unused_features(sample)
        sample = to_device(sample, device)
        sample = postprocess_features(sample)
        return sample

    return chain
