import torch
from torch import nn
from decoder import block


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
                                   dtype=torch.int),
            # Positional Embeddings
            posEmbd = nn.Embedding(config.blockSize,
                                   config.nEmbd,
                                   dtype=torch.int),
            # Hidden layers or Decoder Blocks
            hid = nn.ModuleList([block(config) for _ in range(config.nLayer)]),
            # Layer normalization is applied at the end of each Decoder output
            layerNorm = nn.LayerNorm(config.nEmbd),
            ))
        self.head = nn.Linear(config.nEmbd,
                              config.vocabSize,
                              bias=False,
                              dtype=torch.int)

        self.cnnModel = CnnModel


    def forward(self, Input, Img):
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
        return logits
