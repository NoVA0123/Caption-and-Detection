import torch
from torch import nn
from base_trainer.Transformer.decoder import decoder
from base_trainer.Transformer.encoder import encoder
from base_trainer.Transformer.positional_encoding import positionalencoding
from base_trainer.Transformer.embedding import embeddings


class Transformer(nn.Module):
    def __init__(self,
                 CnnModel,
                 Encoder: encoder,
                 Decoder: decoder,
                 SrcEmbeddings: embeddings,
                 TargetEmbeddings: embeddings,
                 SrcPos: positionalencoding,
                 TargetPos: positionalencoding,
                 DModel: int,
                 VocabSize: int):
        super(Transformer, self).__init__()
        self.cnnModel = CnnModel
        self.encoder = Encoder
        self.decoder = Decoder
        self.srcEmbed = SrcEmbeddings
        self.tgtEmbed = TargetEmbeddings
        self.srcPos = SrcPos
        self.tgtPos = TargetPos
        self.linearLayer = nn.Linear(DModel, VocabSize)

    def encode(self, source):
        source = self.cnnModel(source)
        source = source.type(torch.int)
        source = self.srcEmbed(source)
        source = self.srcPos(source)
        return self.encoder(source)
        
    def decode(self, source, EncoderOutput):
        source = self.tgtEmbed(source)
        source = self.srcPos(source)
        return self.decoder(EncoderOutput, source)

    def projection(self, x):
        x = self.linearLayer(x)
        return torch.softmax(x, dim=-1)

