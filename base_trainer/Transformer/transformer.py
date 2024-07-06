import torch
from torch import nn
from decoder import Decoder
from encoder import Encoder
from positional_encoding import PositionalEncoding
from embedding import Embeddings


class Transformer(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 SrcEmbeddings: Embeddings,
                 TargetEmbeddings: Embeddings,
                 SrcPos: PositionalEncoding,
                 TargetPos: PositionalEncoding,
                 DModel: int,
                 VocabSize: int):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.srcEmbed = SrcEmbeddings
        self.tgt_embed = TargetEmbeddings
        self.src_pos = SrcPos
        self.tgt_pos = TargetPos
        self.linear_layer = nn.Linear(DModel, VocabSize)


    def encode(self, source):
        source = self.srcEmbed(source)
        source = self.src_pos(source)
        return self.encoder(source)
        
    def decoder(self, source, EncoderOutput):
        source = self.tgt_embed(source)
        source = self.src_pos(source)
        return self.decoder(EncoderOutput, source)

    def projection(self, x):
        x = self.linear_layer(x)
        return torch.softmax(x, dim=-1)

