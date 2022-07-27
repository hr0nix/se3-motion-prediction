import torch
import torch.nn as nn

from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.se3_transformer_pytorch import LinearSE3, Fiber


class SE3TransformerModel(nn.Module):
    def __init__(self, feature_dim, num_layers, max_degree, num_heads, num_modes,
                 num_output_timestamps=80, type_0_categorical_num_values=5, type_1_input_feature_dim=2):
        super().__init__()

        if feature_dim % num_heads != 0:
            raise ValueError('feature_dim must be divisible by num_heads')

        self.num_modes = num_modes
        self.num_output_timestamps = num_output_timestamps
        self.feature_dim = feature_dim

        self.preprocess_type0 = nn.Embedding(num_embeddings=type_0_categorical_num_values, embedding_dim=feature_dim)
        self.preprocess_type1 = LinearSE3(
            fiber_in=Fiber({1: type_1_input_feature_dim}), fiber_out=Fiber({1: feature_dim}))
        self.transformer = SE3Transformer(
            dim=feature_dim,
            input_degrees=2,
            out_fiber_dict={0: num_modes, 1: self.num_output_timestamps * num_modes},
            num_degrees=max_degree + 1,
            depth=num_layers,
            heads=num_heads,
            dim_head=feature_dim // num_heads,
        )

    def forward(self, inputs):
        batch_size = inputs['valid'].shape[0]
        num_actors = inputs['valid'].shape[1]

        valid_extra_dim = inputs['valid'][..., None]

        type_0 = inputs['type_0_categorical'] * valid_extra_dim  # To get rid of "-1" before embedding
        type_0_transformed = self.preprocess_type0(type_0).reshape(batch_size, num_actors, self.feature_dim, 1)
        type_1_transformed = self.preprocess_type1({'1': inputs['type_1']})['1']
        result = self.transformer(
            feats={'0': type_0_transformed, '1': type_1_transformed},
            coors=inputs['coords'],
            mask=inputs['valid'],
        )
        mode_log_probs = torch.log_softmax(result['0'].reshape(batch_size, num_actors, self.num_modes), dim=-1)
        mode_means = result['1'].reshape(batch_size, num_actors, self.num_modes, self.num_output_timestamps, 3)
        return {
            'mode_means': mode_means,
            'mode_log_probs': mode_log_probs,
        }
