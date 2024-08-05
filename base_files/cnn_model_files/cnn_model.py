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
    effnetv2s = models.efficientnet_v2_s(pretrained=True)

    # Extracting number of perceptrons from last layer
    NumFeatures = effnetv2s.classifier[1].in_features

    # Freezing every parameter except last
    for params in effnetv2s.parameters():
        params.requires_grad = False

    effnetv2s.classifier[1] = nn.Linear(NumFeatures,
                                        MaxSeqLen * DModel)

    if ExistingPath is not None and os.path.exists(ExistingPath):
        weights = torch.load(ExistingPath)
        effnetv2s.load_state_dict(weights)


    return effnetv2s
