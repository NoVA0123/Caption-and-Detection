import torch
import pandas as pd
import json
from argparse import ArgumentParser
import warnings
from tokenizers import Tokenizer
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.io import read_image
from base_files.transformer_files.dataclass import transformerconfig
from base_files.transformer_files.transformer import transformer
from base_files.cnn_model_files.cnn_model import get_cnn_model


@torch.no_grad()
def CaptionGenerator(JsonPath:str,
                     ImgPath: str):

    device = 'cpu'

    # Use GPU if it is available
    if torch.cuda.is_available():
        device = 'cuda'

    # Use MPS if it is available(Apple devices only)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        

    # Filtering the warnings
    warnings.filterwarnings('ignore')

    null = None

    # Importing json file
    with open (JsonPath, 'r') as f:
        data = json.load(f)

    # Importing tokenizer
    TokenizerPath = data["tokenizer_config"]['tokenizer_load_path']
    tokenizer = Tokenizer.from_file(TokenizerPath)
    

    # Creating a transform image object
    transform = v2.Compose([
        v2.ToDtype(torch.float, scale=True), # Scale the image
        v2.Resize(size=[224, 224]), # Resisze for the model
        v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), # Normalize values
        v2.ToDtype(torch.float) # change it back to float
        ])

    # Reading the image and transforming the image
    img = transform(read_image(ImgPath))


    # Initializing transformer config 
    TrConf = data['transformer_config']
    MaxLen = TrConf['block_size']
    VocabSize = TrConf['vocab_size']
    NumLayers = TrConf['number_layers']
    NumHeads = TrConf['number_heads']
    DModel = TrConf['d_model']

    config = transformerconfig(blockSize=MaxLen,
                               vocabSize=VocabSize,
                               nLayers=NumLayers,
                               nHead=NumHeads,
                               nEmbd=DModel)

    # Initializing model hyper parameters
    ModelConfig = data['model_config']
    BatchSize = ModelConfig['batch_size']

    # Downloading the Cnn model
    CnnConf = data['cnn_model_config']
    ExistingPath = CnnConf['existing_path']
    SpecificDownloadPath = CnnConf['specific_download_path']
    if ExistingPath is not None and SpecificDownloadPath is not None:
        effnetv2s = get_cnn_model(MaxSeqLen=MaxLen,
                                  DModel=DModel,
                                  ExistingPath=ExistingPath,
                                  SpecificDownloadPath=SpecificDownloadPath)
    else:
        effnetv2s = get_cnn_model(MaxSeqLen=MaxLen,
                                  DModel=DModel)


    # Initializing the transformer model
    model = transformer(config=config,
                        CnnModel=effnetv2s)

    # Loading checkpoint
    checkpoint = torch.load(data["saved_model_path"])
    state_dict = checkpoint['model_state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace("module._orig_mod.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)


    '''Creating caption for Image'''
    model.eval()
    NumReturnSequences = 4
    SosToken = torch.tensor([tokenizer.token_to_id('[SOS]')],
                            dtype=torch.long)
    tokens = SosToken.unsqueeze(0).repeat(NumReturnSequences, 1)
    XGen = tokens.to(device) # Sequence Length, DModel
    SampleRng = torch.Generator(device=device)
    SampleRng.manual_seed(42)
    while XGen.size(1) < MaxLen:

        # forwarding the model
        with torch.no_grad():
            logits, _ = model(XGen, img)
            # Take the logits at last position
            logits = logits[:, -1, :]
            # Get the probablities
            probs = F.softmax(logits, dim=-1)
            # TopK sampling
            TopkProbs, TopkIndices = torch.topk(probs, 50, dim=-1)
            # Select a token from topk
            Index = torch.multinomial(TopkProbs, 1, generator=SampleRng)
            # Gather the indices
            xcol = torch.gather(TopkIndices, -1, Index)
            # Append the sequence
            XGen = torch.cat((XGen, xcol), dim=1)

    # Print the text which has been generated
    DecodedValues = []
    for i in range(NumReturnSequences):

        tokens = XGen[i, :MaxLen].tolist()
        decoded = tokenizer.decode(XGen)
        print(decoded)
        DecodedValues.append(decoded)

    return DecodedValues


# Argument parser
def command_line_argument():
    parser = ArgumentParser()
    parser.add_argument('--path', dest='Path', action='append')
    return parser.parse_args()

if __name__ == '__main__':
    Paths = command_line_argument()
    Paths = Paths.Path
    decoded = CaptionGenerator(*Paths)
