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
                 Embeddings: embeddings,
                 Pos: positionalencoding,
                 DModel: int,
                 VocabSize: int):
        super(Transformer, self).__init__()
        self.cnnModel = CnnModel
        self.encoder = Encoder
        self.decoder = Decoder
        self.embed = Embeddings
        self.pos = Pos
        self.linearLayer = nn.Linear(DModel, VocabSize)

    def encode(self, source):
        source = self.cnnModel(source)
        source = source.type(torch.int)
        return self.encoder(source)
        
    def decode(self, source, EncoderOutput):
        source = self.embed(source)
        source = self.pos(source)
        return self.decoder(EncoderOutput, source)

    def projection(self, x):
        x = self.linearLayer(x)
        return torch.softmax(x, dim=-1)

