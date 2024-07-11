import torch
from torch import nn
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class positionalencoding(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self, DModel: int,
                 SeqLen: int) -> None:

        super(positionalencoding, self).__init__()
        self.dModel = DModel
        self.seqLen = SeqLen

        # Matrix of shape(seqlen, dmodel)
        pe = torch.zeros(SeqLen, DModel)

        # Create a vector of shape (seqlen)
        Position = torch.arange(0, SeqLen, dtype=torch.float).unsqueeze(1)

        # Create a vector of shape (dmodel)
        DivTerm = torch.arange(0, DModel, 2).float() * (-math.log(10_000.0)/DModel)
        DivTerm = torch.exp(DivTerm)

        # Apply sine to even positions
        pe[:, 0::2] = torch.sin(Position * DivTerm)
        # Apply cosine to odd positions
        pe[:, 1::2] = torch.cos(Position * DivTerm)
        # Add batch dimention at 0th position
        pe = pe.unsqueeze(0)
        # Register the value
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).to(device).requires_grad_(False)
        return x
