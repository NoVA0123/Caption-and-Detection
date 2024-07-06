import torch
from torch import nn
from multi_head_attention import MultiHeadAttention, CrossMultiHeadAttention
from feed_forward import FeedForward
from layer_normalization import LayerNormalization


class DecoderBlock(nn.Module):
    def __init__(self,
                 DModel: int,
                 Dff: int,
                 Dropout: float,
                 Head: int
                 ):

        super(DecoderBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(Head, 
                                                       DModel)

        self.multi_head_attention2 = CrossMultiHeadAttention(Head, 
                                                       DModel)

        self.feed_forward = FeedForward(DModel,
                                        Dff,
                                        Dropout)
        self.layer_normalization1 = LayerNormalization()
        self.layer_normalization2 = LayerNormalization()
        self.layer_normalization2 = LayerNormalization()

    def forward(self,
                encoder_output,
                y):

        multi_head1 = self.multi_head_attention(y, mask=True)

        # Adding tensor -> LayerNormalization
        AddedTensor = torch.add(y, multi_head1)
        y = self.layer_normalization1(AddedTensor)

        multi_head2= self.multi_head_attention2(y, encoder_output)

        AddedTensor = torch.add(y, multi_head2)
        y = self.layer_normalization2(AddedTensor)

        feed_forward = self.feed_forward(y)
        AddedTensor = torch.add(y, feed_forward)
        y = self.layer_normalization2(AddedTensor)
        return y


class Decoder(nn.Module):
    def __init__(self,
                 layers):
        super(Decoder, self).__init__()
        self.layers = layers

    def forward(self, x, y):
        for layer in self.layers:
            y = layer(x, y)
        return y
        


SequenceLength = 4
BatchSize = 1
InputDim = 10
DModel = 10
Head = 2
Dff = 40
Dropout = 0.2
VocabSize = 10
NumLayers = 6
x = torch.randn(BatchSize, SequenceLength, InputDim)
y = torch.randn(BatchSize, SequenceLength, InputDim)

print("Output before transformer")
print(x)
print(y)

layers = []
for _ in range(6):
    Decoderblock = DecoderBlock(DModel,
                                Dff,
                                Dropout,
                                Head)
    layers.append(Decoderblock)
a = Decoder(layers)

m = a(x, y)
print("\n\n\n\n")
print("Output after transformer")
print(m)
