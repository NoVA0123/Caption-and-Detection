import torch
from torch.cuda import is_available
from base_files.transformer_files.dataclass import transformerconfig
from base_files.transformer_files.transformer import transformer
from base_files.cnn_model_files.cnn_model import get_cnn_model
from base_files.tokenizer_files.tokenizer import get_tokenizer, texttoid
from base_files.dataset_files.json_extracter import caption_extracter
from base_files.dataset_files.image_extracter import imgextracter
import pandas as pd
from torch.utils.data import DataLoader
import json
from tqdm.auto import tqdm
import time
from argparse import ArgumentParser
import warnings


device = 'cpu'

# Use GPU if it is available
if torch.cuda.is_available():
    device = 'cuda'

# Use MPS if it is available(Apple devices only)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'

# Setting seed for reproducability
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

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
    if data['tokenizer_config']['tokenizer_path'] is null:
        tokenizer = get_tokenizer(TrainData)
    else:
        tokenizer = get_tokenizer(TrainData,
                                  data['tokenizer_config']['tokenizer_path'])
    
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
    Lr = ModelConfig['learning_rate']
    Epochs = ModelConfig['epochs']
    
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
                          dataframe=TrainData)

    CaptionData = DataLoader(CaptionDataClass, batch_size=BatchSize)

    # Loading Image data into dataloader
    ImgDataClass = imgextracter(dataframe=TrainData)

    ImgData = DataLoader(ImgDataClass, batch_size=BatchSize)

    # Initializing the transformer model
    model = transformer(config=config,
                        CnnModel=effnetv2s)
    model.to(device) 
    # To compile model and make model faster
    model = torch.compile(model)

    # Initializing optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=Lr,
                                  betas=(0.9, 0.95),
                                  eps=1e-8)

    # Training
    Steps = 0
    for i in tqdm(Epochs):
        SampleImageData = tqdm(ImgData)
        SampleCaptionData = tqdm(CaptionData)

        for img, caption in zip(SampleImageData, SampleCaptionData):
            t0 = time.time()
            DecoderInput = caption['decoder_input'].to(device)
            Label = caption['label'].to(device)
            img = img.to(device)
            optimizer.zero_grad()
            logits, loss = model(DecoderInput, img, label)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0 
            TokensProcessed = BatchSize * MaxLen
            TokensPerSec = TokensProcessed/dt
            print(f"Epoch: {i} | loss: {loss.item()} | norm: {norm} | Process time: {dt*1000:.2f}ms | tok/sec: {TokensPerSec:.2f}")

            Steps += 1
'''
    ModelName = 'caption_model.pt'
    torch.save({
        'epoch': Epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': Steps,
        }, ModelName)'''

# Argument parser
def command_line_argument():
    parser = ArgumentParser()
    parser.add_argument('--path', dest='Path')
    return parser.parse_args()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    JsonPath = command_line_argument()
    train(JsonPath.Path)
