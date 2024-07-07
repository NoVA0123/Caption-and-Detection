import torch
from torch import nn
from base_trainer.Transformer.decoder import decoder, decoderblock
from base_trainer.Transformer.encoder import encoder, encoderblock
from base_trainer.Transformer.positional_encoding import positionalencoding
from base_trainer.Transformer.embedding import embeddings
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
    SrcEmbed = embeddings(DModel, VocabSize)
    TgtEmbed = embeddings(DModel, VocabSize)

    # Creating Positional Encoding layer
    SrcPos = positionalencoding(DModel,
                             MaxSeqLen)
    TgtPos = positionalencoding(DModel,
                             MaxSeqLen)

    # Creating Encoder Block
    EncoderBlocks = []
    for _ in range(NumBlocks):
        Block = encoderblock(DModel,
                                    Dff,
                                    Dropout,
                                    NumHeads)

        EncoderBlocks.append(Block)

    # Creating Decoder Blocks
    DecoderBlocks = []
    for _ in range(NumBlocks):
        Block = decoderblock(DModel,
                             Dff,
                             Dropout,
                             NumHeads)

    # Creating Encoder and Decoder
    Encoder = encoder(nn.ModuleList(EncoderBlocks))
    Decoder = decoder(nn.ModuleList(DecoderBlocks))

    # Creating Transformer
    transformer = Transformer(Encoder,
                              Decoder,
                              SrcEmbed,
                              TgtEmbed,
                              SrcPos,
                              TgtPos,
                              DModel,
                              VocabSize)

    return transformer
