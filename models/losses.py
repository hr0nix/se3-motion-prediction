import torch
from torch.distributions import Normal, Categorical, Independent, MixtureSameFamily


def motion_prediction_loss(predictions, targets, std_dev=1.0):
    batch_size, num_actors, num_modes, num_timestamps, _ = predictions['mode_means'].shape

    predicted_mean = predictions['mode_means']
    mode_log_probs = predictions['mode_log_probs']
    gt_mean = targets['future_coords'].reshape(batch_size, num_actors, num_timestamps, 3)
    relative_gt_mean = gt_mean - targets['coords'].reshape(batch_size, num_actors, 1, 3)

    mode_distribution = Categorical(logits=mode_log_probs)
    position_distribution = MixtureSameFamily(
        mode_distribution,
        Independent(Normal(predicted_mean, std_dev), reinterpreted_batch_ndims=2),
    )

    log_probs = position_distribution.log_prob(relative_gt_mean)
    future_valid_all_ts = torch.all(targets['future_valid'], dim=-1)
    return -torch.mean(log_probs * future_valid_all_ts)  # Average over all actors with all timestamps valid
