import torch
from torch import nn
from torchvision import models
import os


def get_cnn_model(MaxSeqLen:int,
                  DModel:int=512,
                  ExistingPath=None,
                  SpecificDownloadPath=None):

    # If model needs to be downloaded on specifice path
    if SpecificDownloadPath is not None:
        os.environ['TORCH_HOME'] = SpecificDownloadPath

    # Loading the model
    effnetb5 = models.efficientnet_b5(pretrained=True)

    # Extracting number of perceptrons from last layer
    NumFeatures = effnetb5.classifier[1].in_features

    # Freezing every parameter except last
    for params in effnetb5.parameters():
        params.requires_grad = False

    effnetb5.classifier[1] = nn.Linear(NumFeatures,
                                        MaxSeqLen * DModel)

    if ExistingPath is not None and os.path.exists(ExistingPath):
        weights = torch.load(ExistingPath)
        effnetb5.load_state_dict(weights)


    return effnetb5
