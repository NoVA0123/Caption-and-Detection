import torch
import time
import pandas as pd
from torch.utils.data import DataLoader
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser
import warnings
import math
from base_files.transformer_files.dataclass import transformerconfig
from base_files.transformer_files.transformer import transformer
from base_files.cnn_model_files.cnn_model import get_cnn_model
from base_files.tokenizer_files.tokenizer import get_tokenizer, texttoid
from base_files.dataset_files.json_extracter import caption_extracter
from base_files.dataset_files.image_extracter import imgextracter


# Configurig device
print('Loading the device: \n\n')
device = 'cpu'

# Use GPU if it is available
if torch.cuda.is_available():
    device = 'cuda'

# Use MPS if it is available(Apple devices only)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'

print('Device has been loaded!\n\n')
print(f"Current Device: {device}")


# Setting seed for reproducability
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


def get_decay_lr(it:int,
                 WarmupSteps:int,
                 MaxSteps:int,
                 MaxLr:float,
                 MinLr:float):
    # Linear decay for warmup steps
    if it < WarmupSteps:
        return MaxLr * (it + 1) / WarmupSteps
    
    # Constant learning rate
    if it > MaxSteps:
        return MinLr

    # In between we will apply cosine function
    DecayRatio = (it - WarmupSteps) / (MaxSteps - WarmupSteps)
    assert 0 <= DecayRatio <= 1
    Coeff = 0.5 * (1.0 + math.cos(math.pi * DecayRatio))
    return MinLr + Coeff * (MaxLr - MinLr)


# Training the dataset
def train(JsonPath:str):
    null = None
    
    # Loading json
    print("Loading the captions and image paths: \n\n")
    with open (JsonPath, 'r') as f:
        data = json.load(f)

    FilePath = data['file_path']
    TrainJson = FilePath['json_path']['train_json']
    TrainImgPath = FilePath['image_path']['train_path']
    
    # Extracting caption and storing corresponding image path
    TrainData = caption_extracter(TrainJson, TrainImgPath)
    print("Captions and image paths have been loaded into a dataframe. \n\n")

    # Creating a tokenizer
    print('Creating tokenizer: \n\n')
    if data['tokenizer_config']['tokenizer_path'] is null:
        tokenizer = get_tokenizer(TrainData)
    else:
        tokenizer = get_tokenizer(TrainData,
                                  data['tokenizer_config']['tokenizer_path'])

    print("Tokenizer has been created! \n\n")

    # Changing sample size
    TotalSamples = data['dataset_config']['max_sample']
    TrainData = TrainData.sample(TotalSamples,
                                 random_state=1337).reset_index(drop=True)
    
    '''Initializing Config parameters'''

    print('Initializing configuration: \n\n')
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
    Epochs = ModelConfig['epochs']
    
    # Downloading the Cnn model
    print("Loading CNN model: \n\n")
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
    print('Model has been loaded: \n\n')


    # Loading caption data into dataloader
    print('Creating a DataLoader: \n\n')
    CaptionDataClass = texttoid(tokenizer=tokenizer,
                          MaxSeqLen=MaxLen,
                          dataframe=TrainData)

    CaptionData = DataLoader(CaptionDataClass, batch_size=BatchSize)

    # Loading Image data into dataloader
    ImgDataClass = imgextracter(dataframe=TrainData)

    ImgData = DataLoader(ImgDataClass, batch_size=BatchSize)
    print("DataLoader for both Images and captions have been created.\n\n")

    # Initializing the transformer model
    print("Initializing the model: \n\n")
    model = transformer(config=config,
                        CnnModel=effnetv2s)
    model.to(device) 
    # To compile model and make model faster
    print("Model has been created. Now, Compiling the model: \n \n")
    model = torch.compile(model)
    print("Model has been compiled. \n\n")

    '''Initializing optimizer'''
    # Making a decay learning rate
    print("Configuring optimizer with decayed learning rate: \n\n")
    MaxLr = ModelConfig['learning_rate']['max_lr']
    MinLr = MaxLr * 0.1
    WarmupSteps = ModelConfig['learning_rate']['warmup_steps']
    MaxSteps = ModelConfig['learning_rate']['max_steps']

    '''optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=MaxLr,
                                  betas=(0.9, 0.95),
                                  eps=1e-8)'''
    optimizer = model.configure_optimizers(WeightDecay=0.1,
                                           LearningRate=6e-4,
                                           device=device)
    print('Optimizer has been configured.')

    # Creating gradient accumulation step to increase batch size
    TotalBatchSize = 2**19
    assert TotalBatchSize % (BatchSize * MaxLen) == 0, "Make sure the total batch size is divisible by Batch * SeqLen"
    GradAccumSteps = TotalBatchSize // (BatchSize * MaxLen)
    print(f"Total batch size is: {TotalBatchSize} ")
    print(f"-> calculated gradient accumulation steps: {GradAccumSteps}")

    # Training
    GlobalSteps = 0
    for i in tqdm(range(Epochs)):
        IterImgData = iter(ImgData)
        IterCapData = iter(CaptionData)

        LocalSteps = 0
        for _ in range(len(ImgData)//GradAccumSteps):
            t0 = time.time() # Storing time of begining of the step
            # Storing values

            optimizer.zero_grad() # Setting optimizer to zero for every step

            # Accumulated gradient calculation
            for _ in range(GradAccumSteps):
                # Iterating the dataset
                caption = next(IterCapData)
                img = next(IterImgData)
                
                # Storing the values and converting them to device
                DecoderInput = caption['decoder_input'].to(device)
                Label = caption['label'].to(device)
                img = img.to(device)

                '''
                Autocasting to datatypes of model to bfloat16 as it is 4x faster
                than normal float32. It reduces the decimal value.
                '''
                
                with torch.autocast(device_type=device,
                                    dtype=torch.bfloat16):
                    _, loss = model(DecoderInput, img, Label)

                '''
                To calculate Gradient accumulation for larger batches, we need
                to add loss for each micro batch size and scaled it down during
                each step.
                '''
                loss = loss / GradAccumSteps
                loss.backward()

            # Applying norm on gradients to reduce shock of the model
            norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            
            # Decay in learning rate
            lr = get_decay_lr(GlobalSteps,
                              WarmupSteps=WarmupSteps,
                              MaxSteps=MaxSteps,
                              MaxLr=MaxLr,
                              MinLr=MinLr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step() # Applying a backpropogation step
            # Synchronizing GPU and CPU runtime
            torch.cuda.synchronize()
            # Storing output time
            t1 = time.time()
            dt = t1 - t0 
            TokensProcessed = BatchSize * MaxLen
            TokensPerSec = TokensProcessed/dt
            print(f"Epoch: {i} | Steps: {LocalSteps} | loss: {loss.item(): .2f} | lr: {lr: .5e} | norm: {norm: .2f} | Process time: {dt*1000:.2f}ms | tok/sec: {TokensPerSec:.2f}")

            GlobalSteps += 1
            LocalSteps += 1
'''
    ModelName = 'caption_model.pt'
    torch.save({
        'epoch': Epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': GlobalSteps,
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
