import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import inspect


@dataclass
class mArgs:
    dim: int = 512
    nLayers: int = 3
    nHeads: int = 32
    nKVHeads: Optional[int] = None
    VocabSize: int = 30080
    MultipleOf: int = 256
    FFNDIMMULTIPLIER: Optional[float] = None
    NormEps: float = 1e-5

    # Needed for kv cache
    MaxBatchSize: int = 64
    MaxSeqLen: int = 128


def precompute_theta_pos_frequencies(HeadDim: int,
                                    SeqLen: int,
                                    device,
                                    theta: float = 10000.0):
    # Dimension should be even
    assert HeadDim % 2 == 0, "Dimensions must be divisible by 2"
    # formula : theta_i = 10000 ^ (-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Note: Shape = HeadDim / 2
    ThetaNumer = torch.arange(0, HeadDim, 2).float()
    Theta = 1.0 / (theta ** (ThetaNumer/ HeadDim)).to(device)
    # Shape: (Seq Len)
    m = torch.arange(SeqLen, device=device)
    freqs = torch.outer(m, Theta).float()
    FreqComplex = torch.polar(torch.ones_like(freqs), freqs)
    return FreqComplex


def apply_rotary_embeddings(x: torch.Tensor,
                          FreqComplex: torch.Tensor):
    device = FreqComplex.get_device()
    XComp = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    FreqComplex = FreqComplex.unsqueeze(0).unsqueeze(2)
    XRotated = XComp + FreqComplex
    Xout = torch.view_as_real(XRotated)
    Xout = Xout.reshape(*x.shape)
    return Xout.type_as(x).to(device)


class rmsnorm(nn.Module):
    def __init__(self,
                dim: int,
                eps: float = 1e-6):
        super(rmsnorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self,
             x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self,
               x: torch.Tensor):
        return self.weight + self._norm(x.float()).type_as(x)


def repeat_kv(x: torch.Tensor,
             NumRep: int) -> torch.Tensor:
    BatchSize, SeqLen, NKVHeads, HeadDim = x.shape
    if NumRep == 1:
        return x
    else:
        return (
            x[:, :, :, None, :]
            .expand(BatchSize, SeqLen, NKVHeads, NumRep, HeadDim)
            .reshape(BatchSize, SeqLen, NKVHeads * NumRep, HeadDim)
        )


class selfattention(nn.Module):
    def __init__(self,
                args: mArgs):
        super(selfattention, self).__init__()
        self.nKVHeads = args.nHeads if args.nKVHeads is None else args.nKVHeads
        self.nHeadsQ = args.nHeads
        self.nRep = self.nHeadsQ // self.nKVHeads
        self.headDim = args.dim // args.nHeads 

        self.imgLayer = nn.Linear(args.dim, args.nHeads, bias=False)
        self.wQ = nn.Linear(args.dim, args.nHeads * self.headDim, bias=False)
        self.wK = nn.Linear(args.dim, self.nKVHeads * self.headDim, bias=False)
        self.wV = nn.Linear(args.dim, self.nKVHeads * self.headDim, bias=False)
        self.wO = nn.Linear(args.nHeads * self.headDim, args.dim, bias=False)
        self.wO.TRANSFORMER_SCALE_INIT = 1

        self.cacheK = torch.zeros((args.MaxBatchSize,
                                  args.MaxSeqLen,
                                  self.nKVHeads,
                                  self.headDim))
        self.cacheV = torch.zeros((args.MaxBatchSize,
                                  args.MaxSeqLen,
                                  self.nKVHeads,
                                  self.headDim))

    def forward(self,
                x: torch.Tensor,
                FreqComplex: torch.Tensor,
                img:torch.Tensor,
                inference=False,
                StartPos: int = None):
        BatchSize, SeqLen, _ = x.shape
        device = x.get_device()
        img = self.imgLayer(img)
        Xq = self.wQ(x)
        Xk = self.wK(x)
        Xv = self.wV(x)

        Xq = Xq + img
        Xq = Xq.view(BatchSize, SeqLen, self.nHeadsQ, self.headDim)
        Xk = Xk.view(BatchSize, SeqLen, self.nKVHeads, self.headDim)
        values = Xv.view(BatchSize, SeqLen, self.nKVHeads, self.headDim)

        # Apply rotary positional embedding
        Xq = apply_rotary_embeddings(Xq, FreqComplex)
        keys = apply_rotary_embeddings(Xk, FreqComplex)

        # Replace the entry in the cache for this token
        if inference:
            self.cacheK[:BatchSize, StartPos: StartPos + SeqLen] = keys
            self.cacheV[:BatchSize, StartPos: StartPos + SeqLen] = values
    
            # Retrieve the cached values
            keys = self.cacheK[:BatchSize, 0: StartPos + SeqLen].to(device)
            values = self.cacheV[:BatchSize, 0: StartPos + SeqLen].to(device)

            # Repeat the heads of the K and V to reach the number of heads of queries
            keys = repeat_kv(keys, self.nRep)
            values = repeat_kv(values, self.nRep)

        # Transpose all
        Xq = Xq.transpose(1, 2)
        Xk = keys.transpose(1, 2)
        Xv = values.transpose(1, 2)

        # Calculating scores
        scores = F.scaled_dot_product_attention(Xq, Xk, Xv, is_causal=True)

        '''
        Turn this from (B, Heads, 1, Head Dim) -> (B, 1, Dim)
        1. Transpose the matrix's 2nd and 3rd dimension
        2. Values should be contiguous
        3. Convert last 2 dimensions into Dim value
        '''
        Output = (scores.transpose(1, 2).contiguous().view(BatchSize, SeqLen, -1))
        return self.wO(Output)


class feedforward(nn.Module):
    def __init__(self, args: mArgs):
        super(feedforward, self).__init__()

        HiddenDim = 4 * args.dim
        HiddneDim = int(2 * HiddenDim / 3)
        if args.FFNDIMMULTIPLIER is not None:
            HiddenDim = int(args.FFNDIMMULTIPLIER * HiddenDim)

        # Round the Dim to nearest multiple
        HiddenDim = args.MultipleOf * ((HiddenDim + args.MultipleOf - 1) // args.MultipleOf)

        # Feed forward weights
        self.w1 = nn.Linear(args.dim, HiddenDim, bias=False)
        self.w2 = nn.Linear(HiddenDim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, HiddenDim, bias=False)
        self.w2.TRANSFORMER_SCALE_INIT = 1

    def forward(self,
               x: torch.Tensor):
        swish = F.silu(self.w1(x))
        xV = self.w3(x)
        x = swish + xV
        x = self.w2(x)
        return x


class decoderblock(nn.Module):
    def __init__(self,
                args: mArgs):
        super(decoderblock, self).__init__()
        self.nHeads = args.nHeads
        self.dim = args.dim
        self.hDim = args.dim // args.nHeads

        self.attention = selfattention(args)
        self.feedForward = feedforward(args)

        self.attenNorm = rmsnorm(args.dim, eps=args.NormEps)
        self.ffnNorm = rmsnorm(args.dim, eps=args.NormEps)

    def forward(self,
                x: torch.Tensor,
                StartPos: int,
                FreqComplex: torch.Tensor,
                img: torch.Tensor,
                inference=False):
        x = x + self.attention(self.attenNorm(x), FreqComplex, img, inference, StartPos)
        x = x + self.feedForward(self.ffnNorm(x))
        return x


class transformer(nn.Module):
    def __init__(self,
                 args: mArgs,
                 FreqComplex: torch.Tensor,
                 CnnModel) -> None:
        super(transformer, self).__init__()
        assert args.VocabSize != -1, "Vocab size is not set"
        
        self.args = args
        self.vocabSize = args.VocabSize
        self.nLayers = args.nLayers
        self.tokEmbedding = nn.Embedding(self.vocabSize, args.dim)

        self.layers = nn.ModuleList([decoderblock(args) for _ in range(args.nLayers)])

        self.norm = rmsnorm(args.dim, eps=args.NormEps)
        self.output = nn.Linear(args.dim, self.vocabSize, bias=False)

        self.freqsComplex = FreqComplex

        self.convertLayer = nn.Linear(1000, args.dim, bias=False)
        self.tokEmbedding.weight = self.output.weight
        self.cnnModel = CnnModel

        self.apply(self._init_weights)

    def forward(self,
                tokens: torch.Tensor,
                ImgFeat: torch.Tensor,
                Label = None,
                StartPos: int = None):
        # Extract Batch size and sequence length
        inference = False
        BatchSize, SeqLen = tokens.shape
        if Label is None:
            assert SeqLen == 1, "Only one token can be processed at a time"
            inference = True

        x = self.tokEmbedding(tokens)
        ImgFeat = self.cnnModel(ImgFeat)
        ImgFeat = self.convertLayer(ImgFeat)
        ImgFeat = torch.reshape(ImgFeat,
                               (BatchSize, 1, self.args.dim))

        # Pair retrieval from corresponding position
        if inference:
            FreqComplex = self.freqsComplex[StartPos:StartPos + SeqLen]
        else:
            FreqComplex = self.freqsComplex

        # Feeding data to layers
        for layer in self.layers:
            x = layer(x, StartPos, FreqComplex, ImageFeat, inference)
        x = self.norm(x)
        logits = self.output(x).float()
        # Classifying
        if not inference:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   Label.view(-1), ignore_index=-1)
            return logits, loss

        return logits

    def configure_optimizers(self,
                             WeightDecay:float,
                             LearningRate:float,
                             device):
        # Find all the parameters that requires gradients
        Params = {NumParams: p for NumParams, p in self.named_parameters()}
        Params = {NumParams: p for NumParams, p in Params.items() if p.requires_grad}

        '''
        Create optimizer group of weight decay and non decay. Tensors with
        higher dimensions require weight decay to reach optimum value and tensors
        with lower dimensions like bias do not require weight decay.
        '''
        DecayParams = [p for _, p in Params.items() if p.dim() >= 2]
        NonDecayParams = [p for _, p in Params.items() if p.dim() < 2]
        OptimGroups = [
                {'params': DecayParams, 'weight_decay': WeightDecay},
                {'params': NonDecayParams, 'weight_decay': 0.0}
                ]

        # To find number of decay and non decay parameters
        NumDecayParams = sum(p.numel() for p in DecayParams)
        NumNonDecayParams = sum(p.numel() for p in NonDecayParams)
        print(f"Number of decaying parameter tensors: {len(DecayParams)}, with {NumDecayParams} parameters")
        print(f"Number of non decaying parameter tensors: {len(NonDecayParams)}, with {NumNonDecayParams} parameters")

        # Check fused is available or not
        if 'cuda' in device:
            FusedAvailable = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            UseFused = FusedAvailable 
        else:
            UseFused = False
        print(f'Using fused AdamW: {UseFused}')
        # Configuring optimizer
        Optimizer = torch.optim.AdamW(OptimGroups,
                                      lr=LearningRate,
                                      betas=(0.9, 0.95),
                                      eps=1e-8,
                                      fused=UseFused)
        return Optimizer
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tokEmbedding.weight.numel()
        return n_params
