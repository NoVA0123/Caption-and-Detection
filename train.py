import torch
import pandas as pd
import tokenizers
from base_trainer.Tokenizer.tokenizer_model import tokenizer_creator, convert
from config import get_config, get_weights_file_path
from base_trainer.TextPreprocess.dataset_creator import *
from config import get_config, get_weights_file_path
from base_trainer.vision_model.vision_model import vision_model
from model import build_transformer
from base_trainer.TextPreprocess.dataset_creator import create_dataset, preprocess, length_finder


# Get vision model
def get_vision_model(VocabSize: int,
                     ModelPath=None,
                     SpecifiedPath=None):

    model = vision_model(VocabSize, ModelPath, SpecifiedPath)
    return model


def train(TrainPath: str,
          ValPath: str,
          TrainImgPath: str,
          ValImgPath: str,
          TokenizerPath: str,
          CnnModlPath=None,
          SpecifiedPath=None):

    # Creating the dataset
    TrainData, ValData = create_dataset(TrainPath,
                                        ValPath,
                                        TrainImgPath,
                                        ValImgPath)

    # Preprocessing the text
    TrainData['caption'] = TrainData['caption'].apply(preprocess)
    ValData['caption'] = ValData['caption'].apply(preprocess)


    # Finding the maximum length of the sentence
    MaxLen = length_finder(TrainData, ValData)


    # Getting the tokenizer
    tokenizer = tokenizer_creator(TrainData,
                                  TokenizerPath)

    # Converting captions into tokens
    # On training  data
    TrainCaption = convert(TrainData,
                           tokenizer,
                           MaxLen)

    # On validation data
    ValCaption = convert(ValData,
                           tokenizer,
                           MaxLen)


