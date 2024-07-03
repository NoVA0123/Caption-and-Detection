import torch
from torch import nn


class FeedForward(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self,
                 DModel: int,
                 Dff: int,
                 Dropout: float):
        '''Normalization will work in 2 steps:
           x(last dim = d_model) -> x(last dim = dff) -> x(last dim = d_model)'''
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(DModel, Dff)
        self.dropout = nn.Dropout(Dropout)
        self.linear2 = nn.Linear(Dff, DModel)


    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
