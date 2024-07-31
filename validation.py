import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
from argparse import ArgumentParser
import warnings
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import torch.multiprocessing as mp
from base_files.transformer_files.dataclass import transformerconfig
from base_files.transformer_files.transformer import transformer
from base_files.cnn_model_files.cnn_model import get_cnn_model
from base_files.tokenizer_files.tokenizer import get_tokenizer, texttoid
from base_files.dataset_files.json_extracter import caption_extracter
from base_files.dataset_files.image_extracter import imgextracter


# Setting the seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


@torch.no_grad()
def validation(JsonPath:str):

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


    # Importing the file
    with open (JsonPath, 'r') as f:
        data = json.load(f)

    FilePath = data['file_path']
    ValJson = FilePath['json_path']['validation_json']
    ValImgPath = FilePath['image_path']['validation_path']
    
    # Extracting caption and storing corresponding image path
    ValData = caption_extracter(ValJson, ValImgPath)

    # Loading the tokenizer
    TokenizerPath = data["tokenizer_config"]['tokenizer_load_path']
    tokenizer = Tokenizer.from_file(TokenizerPath)
    


    '''Initializing Config parameters'''

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

    # Loading caption data into dataloader
    CaptionDataClass = texttoid(tokenizer=tokenizer,
                          MaxSeqLen=MaxLen,
                          dataframe=ValData)

    CaptionData = DataLoader(CaptionDataClass,
                             batch_size=BatchSize)

    # Loading Image data into dataloader
    ImgDataClass = imgextracter(dataframe=ValData)

    ImgData = DataLoader(ImgDataClass,
                         batch_size=BatchSize)

    # Initializing the transformer model
    model = transformer(config=config,
                        CnnModel=effnetv2s)

    # Loading checkpoint
    checkpoint = torch.load(data["saved_model_path"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Setting up for validation

    NumCorrectNorm = 0
    NumCorrect = 0
    NumTotal = 0

    # Validation begins here
    for img, caption in zip(ImgData, CaptionData):

        DecoderInput = caption['decoder_input'].to(device)
        Label = caption['label'].to(device)
        img = img.to(device)

        # get logits
        logits, loss = model(DecoderInput, img)
        ShiftLogits = (logits[:, :-1, :]).contiguous()
        ShiftInput = (DecoderInput[:, :-1, :]).contiguous()

