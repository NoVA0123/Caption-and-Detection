import torch
from torch import nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_normalization import LayerNormalization


class EncoderBlock(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self,
                 DModel: int,
                 Dff: int,
                 Dropout: float,
                 Head: int
                 ):
        super(EncoderBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(Head, # No masking needed
                                                       DModel)
        self.feed_forward = FeedForward(DModel,
                                        Dff,
                                        Dropout)
        self.layer_normalization1 = LayerNormalization()
        self.layer_normalization2 = LayerNormalization()


    def forward(self, x):
        # Saving  orignal value
        Value = x
        x = self.multi_head_attention(x)
        
        # Adding tensor -> LayerNormalization
        AddedTensor = torch.add(x, Value)
        x = self.layer_normalization1(AddedTensor)
        Value = x
        
        # Doing the above coded 2nd time
        x = self.feed_forward(x)
        AddedTensor = torch.add(x, Value)
        Value = self.layer_normalization2(AddedTensor)
        return Value

    pass


class Encoder(nn.Module):
    def __init__(self,
                 layers):
        super(Encoder, self).__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
