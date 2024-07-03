import torch
from torch import nn


class LayerNormalization(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self, epsilon: float = 10**-4):
        super(LayerNormalization, self).__init__()
        self.eps = epsilon # Normalizer denominator is never 0
        self.alpha = nn.Parameter(torch.ones(1)) # Weight
        self.bias = nn.Parameter(torch.zeros(1)) # Bias


    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # Mean
        std = x.std(dim=-1, keepdim=True) # Deviation
        return self.alpha * ((x - mean) / (std + self.eps)) + self.bias
