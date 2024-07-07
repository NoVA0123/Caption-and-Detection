import torch
from self_attention import selfattention
from torch import nn

class multiheadattention(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self, heads: int, DModel: int):
        assert DModel % heads == 0
        super(multiheadattention, self).__init__()
        self.heads = heads
        self.headDim = DModel // heads
        self.attention = selfattention()
        self.qkvLayer = nn.Linear(DModel, 3*DModel)
        self.linearLayer = nn.Linear(DModel, DModel)
        

    def forward(self, InputQuery, mask=False):
        # Getting shape of input
        BatchSize, SequenceLength, _ = InputQuery.shape
        x = self.qkvLayer(InputQuery)
        x = x.reshape(BatchSize,
                      SequenceLength,
                      self.heads,
                      3 *self.headDim)
        x = x.permute(0, 2, 1, 3)
        Query, Key, Value = x.chunk(3, dim=-1)
        AttentionScore = self.attention(Query, Key, Value, mask=mask)
        AttentionScore = AttentionScore.reshape(BatchSize,
                                                SequenceLength,
                                                self.heads*self.headDim)
        Output = self.linearLayer(AttentionScore)
        return Output


    pass


class crossmultiheadattention(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self, heads: int, DModel: int):
        assert DModel % heads == 0
        super(crossmultiheadattention, self).__init__()
        self.heads = heads
        self.headDim = DModel // heads
        self.attention = selfattention()
        self.qkLayer = nn.Linear(DModel, 2*DModel)
        self.vLayer = nn.Linear(DModel, DModel)
        self.linearLayer = nn.Linear(DModel, DModel)
        

    def forward(self, InputQuery, EncodingOutput,  mask=False):

        # Getting shape of input
        BatchSize, SequenceLength, _ = EncodingOutput.shape
        x = self.qkLayer(EncodingOutput)
        x = x.reshape(BatchSize,
                      SequenceLength,
                      self.heads,
                      2 *self.headDim)
        x = x.permute(0, 2, 1, 3)
        Value = self.vLayer(InputQuery)
        Value = Value.reshape(BatchSize,
                      SequenceLength,
                      self.heads,
                      self.headDim)
        Value = Value.permute(0, 2, 1, 3)
        Query, Key = x.chunk(2, dim=-1)
        AttentionScore = self.attention(Query, Key, Value, mask=mask)
        AttentionScore = AttentionScore.reshape(BatchSize,
                                                SequenceLength,
                                                self.heads*self.headDim)
        Output = self.linearLayer(AttentionScore)
        return Output


    pass
