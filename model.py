import torch
from torch.cuda import is_available
from base_files.transformer_files.dataclass import transformerconfig
from base_files.transformer_files.transformer import transformer
from base_files.cnn_model_files.cnn_model import get_cnn_model
from base_files.tokenizer_files.tokenizer import get_tokenizer, texttoid
from base_files.dataset_files.json_extracter import caption_extracter
from base_files.dataset_files.image_extracter import imgextracter
import pandas as pd
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.utils.data import DataLoader
import json
from tqdm.auto import tqdm


device = 'cpu'

# Use GPU if it is available
if torch.cuda.is_available():
    device = 'cuda'

# Use MPS if it is available(Apple devices only)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'


def train(JsonPath:str):
    null = None
    
    # Loading json
    with open (JsonPath, 'r') as f:
        data = json.load(f)

    FilePath = data['file_path']
    TrainJson = FilePath['json_path']['train_json']
    TrainImgPath = FilePath['image_path']['train_path']
    
    # Extracting caption and storing corresponding image path
    TrainData = caption_extracter(TrainJson, TrainImgPath)

    # Creating a tokenizer
    if data['tokenizer_config'] is null:
        tokenizer = get_tokenizer(TrainData)
    else:
        tokenizer = get_tokenizer(TrainData, data['tokenizer_config'])
    
    '''Initializing Hyper Parameters'''
    # Getting vocab size
    VocabSize = tokenizer.get_vocab_size()

    # Getting Max Sequence Length
    MaxLen = 0
    for i in tqdm(TrainData['caption'].tolist()):
        m = i.split()
        if len(m) > MaxLen:
            MaxLen = len(m)

    # Initializing transformer config 
    TrConf = data['transformer_config']
    NumLayers = TrConf['number_layers']
    NumHeads = TrConf['number_heads']
    DModel = TrConf['d_model']

    config = transformerconfig(blockSize=MaxLen,
                               vocabSize=VocabSize,
                               nLayers=NumLayers,
                               nHead=NumHeads,
                               nEmbd=DModel)
    
    # Downloading the Cnn model
    CnnConf = data['cnn_model_config']
    ExistingPath = CnnConf['existing_path']
    SpecificDownloadPath = CnnConf['specific_download_path']
    effnetv2s = get_cnn_model(MaxSeqLen=MaxLen,
                              DModel=DModel,
                              ExistingPath=ExistingPath,
                              SpecificDownloadPath=SpecificDownloadPath)
    
    # Initializing the transformer model
    model = transformer(config=config,
                        CnnModel=effnetv2s)
    
    # Initializing model hyper parameters
    ModelConfig = data['model_config']
    BatchSize = ModelConfig['batch_size']
    Lr = ModelConfig['learning_rate']
    Epochs = ModelConfig['epochs']

    # Loading caption data into dataloader
    CaptionDataClass = texttoid(tokenizer=tokenizer,
                          MaxSeqLen=MaxLen,
                          dataframe=TrainData)

    CaptionData = DataLoader(CaptionDataClass, batch_size=BatchSize)

    # Loading Image data into dataloader
    ImgDataClass = imgextracter(dataframe=TrainData)

    ImgData = DataLoader(ImgDataClass, batch_size=BatchSize)
