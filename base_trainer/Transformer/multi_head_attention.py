import torch
from self_attention import SelfAttention
from torch import nn

class MultiHeadAttention(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self, heads: int, DModel: int):
        assert DModel % heads == 0
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.head_dim = DModel // heads
        self.attention = SelfAttention()
        self.qkv_layer = nn.Linear(DModel, 3*DModel)
        self.linear_layer = nn.Linear(DModel, DModel)
        

    def forward(self, InputQuery, mask=False):
        # Getting shape of input
        BatchSize, SequenceLength, _ = InputQuery.shape
        x = self.qkv_layer(InputQuery)
        x = x.reshape(BatchSize,
                      SequenceLength,
                      self.heads,
                      3 *self.head_dim)
        x = x.permute(0, 2, 1, 3)
        Query, Key, Value = x.chunk(3, dim=-1)
        AttentionScore = self.attention(Query, Key, Value, mask=mask)
        AttentionScore = AttentionScore.reshape(BatchSize,
                                                SequenceLength,
                                                self.heads*self.head_dim)
        Output = self.linear_layer(AttentionScore)
        return Output


    pass


class CrossMultiHeadAttention(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self, heads: int, DModel: int):
        assert DModel % heads == 0
        super(CrossMultiHeadAttention, self).__init__()
        self.heads = heads
        self.head_dim = DModel // heads
        self.attention = SelfAttention()
        self.qk_layer = nn.Linear(DModel, 2*DModel)
        self.v_layer = nn.Linear(DModel, DModel)
        self.linear_layer = nn.Linear(DModel, DModel)
        

    def forward(self, InputQuery, encoding_output,  mask=False):

        # Getting shape of input
        BatchSize, SequenceLength, _ = encoding_output.shape
        x = self.qk_layer(encoding_output)
        x = x.reshape(BatchSize,
                      SequenceLength,
                      self.heads,
                      2 *self.head_dim)
        x = x.permute(0, 2, 1, 3)
        Value = self.v_layer(InputQuery)
        Value = Value.reshape(BatchSize,
                      SequenceLength,
                      self.heads,
                      self.head_dim)
        Value = Value.permute(0, 2, 1, 3)
        Query, Key = x.chunk(2, dim=-1)
        AttentionScore = self.attention(Query, Key, Value, mask=mask)
        AttentionScore = AttentionScore.reshape(BatchSize,
                                                SequenceLength,
                                                self.heads*self.head_dim)
        Output = self.linear_layer(AttentionScore)
        return Output


    pass
