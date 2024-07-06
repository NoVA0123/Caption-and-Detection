import torch
from torch import nn
from base_trainer.Transformer.decoder import Decoder, DecoderBlock
from base_trainer.Transformer.encoder import Encoder, EncoderBlock
from base_trainer.Transformer.positional_encoding import PositionalEncoding
from base_trainer.Transformer.embedding import Embeddings
from base_trainer.Transformer.transformer import Transformer


# Building the model
def build_transformer(VocabSize: int,
                      MaxSeqLen: int,
                      DModel: int=512,
                      NumBlocks: int=6,
                      NumHeads: int=8,
                      Dropout: float=0.1,
                      Dff: int=2048) -> Transformer:

    # Creating Embedding layer
    SrcEmbed = Embeddings(DModel, VocabSize)
    TgtEmbed = Embeddings(DModel, VocabSize)

    # Creating Positional Encoding layer
    SrcPos = PositionalEncoding(DModel,
                             MaxSeqLen)
    TgtPos = PositionalEncoding(DModel,
                             MaxSeqLen)

    # Creating Encoder Block
    EncoderBlocks = []
    for _ in range(NumBlocks):
        Block = EncoderBlock(DModel,
                                    Dff,
                                    Dropout,
                                    NumHeads)

        EncoderBlocks.append(Block)

    # Creating Decoder Blocks
    DecoderBlocks = []
    for _ in range(NumBlocks):
        Block = DecoderBlock(DModel,
                             Dff,
                             Dropout,
                             NumHeads)

    # Creating Encoder and Decoder
    encoder = Encoder(nn.ModuleList(EncoderBlocks))
    decoder = Decoder(nn.ModuleList(DecoderBlocks))

    # Creating Transformer
    transformer = Transformer(encoder,
                              decoder,
                              SrcEmbed,
                              TgtEmbed,
                              SrcPos,
                              TgtPos,
                              DModel,
                              VocabSize)

    return transformer
