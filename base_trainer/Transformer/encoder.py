import torch
from torch import nn
from multi_head_attention import multiheadattention
from feed_forward import feedforward
from layer_normalization import layernormalization


class encoderblock(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self,
                 DModel: int,
                 Dff: int,
                 Dropout: float,
                 Head: int
                 ):
        super(encoderblock, self).__init__()
        self.multiHeadAttention = multiheadattention(Head, # No masking needed
                                                       DModel)
        self.feedForward = feedforward(DModel,
                                        Dff,
                                        Dropout)
        self.layerNormalization1 = layernormalization()
        self.layerNormalization2 = layernormalization()


    def forward(self, x):
        # Saving  orignal value
        Value = x
        x = self.multiHeadAttention(x)
        
        # Adding tensor -> LayerNormalization
        AddedTensor = torch.add(x, Value)
        x = self.layerNormalization1(AddedTensor)
        Value = x
        
        # Doing the above coded 2nd time
        x = self.feedForward(x)
        AddedTensor = torch.add(x, Value)
        Value = self.layerNormalization2(AddedTensor)
        return Value

    pass


class encoder(nn.Module):
    def __init__(self,
                 layers):
        super(encoder, self).__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
