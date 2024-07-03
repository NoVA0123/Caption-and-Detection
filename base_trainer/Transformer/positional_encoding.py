import torch
from torch import nn

class PositionalEncoding(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self, DModel: int,
                 SeqLen: int) -> None:

        super(PositionalEncoding, self).__init__()
        self.d_model = DModel
        self.seq_len = SeqLen


    def forward(self, x):
        EvenPos = torch.arange(0, self.d_model, 2).float()
        Denom = torch.pow(10_000, EvenPos/self.d_model)

        Positions = torch.arange(self.seq_len).reshape(self.seq_len, 1)
        EvenPe = torch.sin(Positions/Denom)
        OddPe = torch.sin(Positions/Denom)

        stacking = torch.stack([EvenPe, OddPe], dim=2)
        Pe = torch.flatten(stacking, start_dim=1, end_dim=2)
        x = x + Pe.requires_grad_(False)
        return x  
