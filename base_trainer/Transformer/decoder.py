import torch
from torch import nn
from multi_head_attention import multiheadattention, crossmultiheadattention
from feed_forward import feedforward
from layer_normalization import layernormalization


class decoderblock(nn.Module):
    def __init__(self,
                 DModel: int,
                 Dff: int,
                 Dropout: float,
                 Head: int
                 ):

        super(decoderblock, self).__init__()
        self.multiHeadAttention1 = multiheadattention(Head, 
                                                       DModel)

        self.multiHeadAttention2 = crossmultiheadattention(Head, 
                                                       DModel)

        self.feedForward = feedforward(DModel,
                                        Dff,
                                        Dropout)
        self.layerNormalization1 = layernormalization()
        self.layerNormalization2 = layernormalization()
        self.layerNormalization3 = layernormalization()

    def forward(self,
                EncoderOutput,
                y):

        MultiHead1 = self.multiHeadAttention1(y, mask=True)

        # Adding tensor -> LayerNormalization
        AddedTensor = torch.add(y, MultiHead1)
        y = self.layerNormalization1(AddedTensor)

        MultiHead2= self.multiHeadAttention2(y, EncoderOutput)

        AddedTensor = torch.add(y, MultiHead2)
        y = self.layerNormalization2(AddedTensor)

        FeedForward = self.feedForward(y)
        AddedTensor = torch.add(y, FeedForward)
        y = self.layerNormalization3(AddedTensor)
        return y


class decoder(nn.Module):
    def __init__(self,
                 layers):
        super(decoder, self).__init__()
        self.layers = layers

    def forward(self, x, y):
        for layer in self.layers:
            y = layer(x, y)
        return y
