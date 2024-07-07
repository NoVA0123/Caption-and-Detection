import torch
from torch import nn
from math import sqrt


class selfattention(nn.Module):
    '''https://arxiv.org/pdf/1706.03762'''
    def __init__(self):
        super(selfattention, self).__init__()
        '''Q, K, V came from same embedding but different layer'''

    def scale(self, Q, K):
        '''Transpose K then multiply with Q and divide by Dimensions
            This helps to reduce variance'''
        DimensionsScaler = Q.size(dim=-1) # This is basically self.head_dim
        # According to paper Key tensor should be transposed
        KTrans = torch.transpose(K, -2, -1)
        # Matrix multiplication of Query and Key Tensor
        MatMul = torch.matmul(Q, KTrans)
        return MatMul/sqrt(DimensionsScaler) # Scaling it Down

    def forward(self, Q, K, V, mask=False):
        Scaled = self.scale(Q, K) # Calling scale funciton on Query and key tensor
        # If values are small mask it with -infinity
        # This will help softmax to give 0 score
        if mask:
            Masked = torch.full(Scaled.size(), float('-inf'))
            Masked = torch.triu(Masked, diagonal=1)
            Scaled += Masked
        # Applying softmax using last dimension
        Softmaxed = nn.functional.softmax(Scaled, dim=-1)
        return torch.matmul(Softmaxed, V)

    pass
