import torch
from torch import nn
from base_files.transformer_files.decoder import block
from torch.nn import functional as F


class transformer(nn.Module):

    def __init__(self, config, CnnModel):
        super(transformer, self).__init__()
        self.config = config # Transformer config
        '''
        We need to insert modules to the transformer. To do it ModuleDict will
        be used to insert the modules
        '''
        self.transformer = nn.ModuleDict(dict(
            # Token Embeddings at input stage
            tokEmbd = nn.Embedding(config.vocabSize,
                                   config.nEmbd,
                                   dtype=torch.float),
            # Positional Embeddings
            posEmbd = nn.Embedding(config.blockSize,
                                   config.nEmbd,
                                   dtype=torch.float),
            # Hidden layers or Decoder Blocks
            hid = nn.ModuleList([block(config) for _ in range(config.nLayers)]),
            # Layer normalization is applied at the end of each Decoder output
            layerNorm = nn.LayerNorm(config.nEmbd),
            ))
        self.head = nn.Linear(config.nEmbd,
                              config.vocabSize,
                              bias=False,
                              dtype=torch.float)

        # Cnn Model
        self.cnnModel = CnnModel

        # Pointing final Linear projection weights to token embedding weights
        self.transformer.tokEmbd.weight = self.head.weight

        # Initializing weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        '''
        This function is to initialize weights
        '''
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'TRANSFORMER_SCALE_INIT'):
                std *= (2 * self.config.nLayers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0., std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0., std=0.02)

    def forward(self, Input, Img, Target=None):
        # Input is of shape (BatchSize, SeqLen)
        BatchSize, SeqLen = Input.size()
        assert SeqLen <= self.config.blockSize, f"Cannot pass the sequence to the model, Error: length {SeqLen} is greater than the block size parameter for the model"

        # Applying embeddings and tokenization
        Pos = torch.arange(0, SeqLen, dtype=torch.int, device=Input.device)
        PosEmbd = self.transformer.posEmbd(Pos)
        TokEmbd = self.transformer.tokEmbd(Input)

        # Adding both the embeddings
        x = PosEmbd + TokEmbd
        

        # applying decoder block
        for block in self.transformer.hid:
            x = block(x)

        # forward the final layernorm
        x = self.transformer.layerNorm(x)

        # Passing image through Cnn Model
        CnnOutput = self.cnnModel(Img)
        CnnOutput = torch.reshape(CnnOutput,
                                  (BatchSize, SeqLen, self.config.nEmbd)) 
        # Adding Cnn model output before sending it to decoder
        x = x + CnnOutput

        # Classifying
        logits = self.head(x)

        # Calculating loss
        loss = None
        if Target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   Target.view(-1))
        return logits, loss
