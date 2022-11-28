import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch_functions import NormalizationLayer

class RegionLearner(nn.Module):
    def __init__(self, token_dim, num_region=8):
        super(RegionLearner, self).__init__()

        self.num_region = num_region
        self.token_dim= token_dim
        self.normalization_layer = NormalizationLayer(
            normalize_scale=4.0, learn_scale=True)

        self.spatial_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.spatial_att = nn.Conv2d(in_channels=token_dim,
                        out_channels=self.num_region, # each channel used as att map for capturing one region
                        kernel_size=3,
                        padding=1
                        )

    def forward(self, x):
        N, HW, C = x.shape
        H, W = int(HW**0.5), int(HW**0.5)
        # x = x.view(N, C, H, W)
        x = x.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        # x [B, C, H, W]
        B = x.size(0)
        region_mask = self.spatial_att(x) # [B, S, H, W]

        learned_region_list = []
        for s in range(self.num_region):
            learned_region = x * region_mask[:,s,...].unsqueeze(1) # [B, C, H, W] * [B, 1, H, W] --> [B, C, H, W]
            # learned_region_list.append(self.spatial_pooling(learned_region).reshape(B, self.token_dim)) # [B, C, H, W] --> [B, C, 1, 1]
            learned_region = self.spatial_pooling(learned_region).reshape(B, self.token_dim)
            learned_region = self.normalization_layer(learned_region)
            learned_region_list.append(learned_region)

        #  learned_region_list [B, C, 1]
        # print(learned_region_list[0].size())
        learned_regions = torch.stack(learned_region_list, dim=-1) # [B, C, S]
        learned_regions = learned_regions.view(B, -1)
        return learned_regions, region_mask # [B, C, S],

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
