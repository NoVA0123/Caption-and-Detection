import torch
from torch import nn
import pandas as pd
import tokenizers
from base_trainer.Tokenizer.tokenizer_model import tokenizer_creator, convert
from base_trainer.TextPreprocess.dataset_creator import *
from config import get_config, get_weights_file_path
from base_trainer.vision_model.vision_model import vision_model
from model import build_transformer
from base_trainer.TextPreprocess.dataset_creator import create_dataset, preprocess, length_finder
from pathlib import Path
import numpy
from torchvision.transforms import v2
from torchvision.io import read_image
from base_trainer.vision_model.vision_model import vision_model
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from argparse import ArgumentParser
import warnings


# Get vision model
class loadimg(torch.utils.data.Dataset):
    def __init__(self, DataFrame):
        self.dataFrame = DataFrame
        self.transform = v2.Compose([
            v2.ToDtype(torch.float16, scale=True), # turns images into float16
            v2.Resize(size=[224, 224]), # Resizes the image
            v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), # Normalize values
            v2.ToDtype(torch.half)
            ])

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, Index):
        # 'image_path' column has path to images
        Img = self.dataFrame['image_path'][Index] # Store path
        Img = read_image(Img) # Read the images using paths
        return self.transform(Img) # Return the transformed image


def train(TrainPath: str,
          ValPath: str,
          TrainImgPath: str,
          ValImgPath: str,
          TokenizerPath: str,
          CnnModelPath=None,
          SpecifiedPath=None,
          Epochs: int=5):

    # Creating the dataset
    print('Creating Data set :-')
    TrainData, ValData = create_dataset(TrainPath,
                                        ValPath,
                                        TrainImgPath,
                                        ValImgPath)
    print('Dataset have been created. \n \n')

    # Preprocessing the text
    TrainData['caption'] = TrainData['caption'].apply(preprocess)
    ValData['caption'] = ValData['caption'].apply(preprocess)


    # Finding the maximum length of the sentence
    print('Finding the maximum length :-')
    MaxLen = length_finder(TrainData, ValData)

    print('Maximum length has been found! \n \n')


    # Getting the tokenizer
    print('Tokenizer is being created :-')
    tokenizer = tokenizer_creator(TrainData,
                                  TokenizerPath)
    print('Tokenizer has been created. \n \n')

    # Initializing vocab size
    VocabSize = tokenizer.get_vocab_size()


    # Cnn transfer learning model
    print("Loading the Cnn model :-")
    effnetv2s = vision_model(VocabSize, CnnModelPath, SpecifiedPath)
    print("Cnn model has been loaded!")

    '''
    Image Augmentation coming soon
    '''
    # Getting the config file
    config = get_config(MaxLen, Epochs)


    # Taking 70,000 samples due to memory constraints
    TrainData = TrainData.sample(70_000, random_state=42).reset_index(drop=True)


    # Converting captions into tokens
    # On training  data
    print('Converting into captions :-')
    TrainCaption = convert(TrainData,
                           tokenizer,
                           MaxLen)

    # On validation data
    ValCaption = convert(ValData,
                         tokenizer,
                         MaxLen)
    print('Dataset has been converted into captions. \n \n')


    # Loading the data
    # Converting caption into batches of data
    # Train Caption
    TrainCaption = DataLoader(TrainCaption, batch_size=config['batch_size'])

    # Val Caption
    ValCaption = DataLoader(ValCaption, batch_size=1)

    # Loading Images
    TrainImg = loadimg(TrainData)
    TrainImg = DataLoader(TrainImg, batch_size=config['batch_size'])


    # Defining the device, if there is cuda use cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current Device: {device}')
    # Setting up the path for saving weights
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    
    # Getting the model
    model = build_transformer(effnetv2s, VocabSize, MaxLen).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Optimzer
    Optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 eps=1e-9)


    # Initializing training method
    InitialEpoch = 0
    GlobalStep = 0
    if config['preload']:
        ModelFilename = get_weights_file_path(config, config['preload'])
        print(f'Preloading the model {ModelFilename}')
        State = torch.load(ModelFilename)
        InitialEpoch = State['epoch'] + 1
        Optimizer.load_state_dict(State['optimizer_state_dict'])
        GlobalStep = State['global_step']

    # There are multiple words therefore, I will use Cross Entropy
    LossFn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    
    # Training begins from here
    for epoch in range(InitialEpoch, config['num_epochs']):

        model.train()
        DecodeBatch = tqdm(TrainCaption)
        TrainImgBatch = tqdm(TrainImg)

        for img, batch in zip(TrainImgBatch, DecodeBatch):

            Img = img.to(device)
            DecoderInput = batch['decoder_input'].to(device)

            # Run the tensors throught the transformers
            EncoderOutput = model.encode(Img)
            DecoderOutput = model.decode(DecoderInput, EncoderOutput)
            ProjOutput = model.projection(DecoderOutput)

            label = batch['label'].to(device)


            # calculating loss
            loss = LossFn(ProjOutput.view(-1, VocabSize), label.view(-1))
            DecodeBatch.set_postfix({f'loss': '{loss.item(): 6.3f}'})

            # Log the loss in tensorboard
            writer.add_scalar('train loss', loss.item(), GlobalStep)
            writer.flush()


            # Backpropogation
            loss.backward()

            # Update the weights
            Optimizer.step()
            Optimizer.zero_grad()

            GlobalStep += 1


        # Save the model at the end of every epoch
        ModelFilename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': Optimizer.state_dict(),
            'global_step': GlobalStep
            }, ModelFilename)



def get_command_line_arguments():
    parser = ArgumentParser()
    # Specifies required paths
    parser.add_argument("--paths",
                        dest='Paths',
                        required=True,
                        action='append',
                        help='Specify the paths')
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = get_command_line_arguments()
    args = args.Paths
    train(*args)
