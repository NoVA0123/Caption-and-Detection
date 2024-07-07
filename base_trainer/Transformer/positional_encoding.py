import torch
from torch import nn

class positionalencoding(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self, DModel: int,
                 SeqLen: int) -> None:

        super(positionalencoding, self).__init__()
        self.dModel = DModel
        self.seqLen = SeqLen


    def forward(self, x):
        EvenPos = torch.arange(0, self.dModel, 2).float()
        Denom = torch.pow(10_000, EvenPos/self.dModel)

        Positions = torch.arange(self.seqLen).reshape(self.seqLen, 1)
        EvenPe = torch.sin(Positions/Denom)
        OddPe = torch.sin(Positions/Denom)

        stacking = torch.stack([EvenPe, OddPe], dim=2)
        Pe = torch.flatten(stacking, start_dim=1, end_dim=2)
        x = x + Pe.requires_grad_(False)
        return x  
