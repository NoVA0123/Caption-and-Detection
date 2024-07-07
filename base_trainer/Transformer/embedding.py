from torch import nn
from math import sqrt

class embeddings(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self, DModel: int,
                 VocabSize: int) -> None:

        super(embeddings, self).__init__()
        '''We will turn vocab size input into
           D model size output.'''
        self.dModel = DModel # Output Embeddings
        self.embeddings = nn.Embedding(VocabSize, DModel)


    def forward(self, x):
        # According to paper embeddings were multiplied by root of output size
        return self.embeddings(x) * sqrt(self.dModel)
