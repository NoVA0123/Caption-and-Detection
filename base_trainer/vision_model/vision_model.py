import os
import torch
from torchvision import models
from torch import nn

# funciton for downloading and modifying the model
def vision_model(VocabSize: int,
                 ModelPath: str):

    # If model exists load it
    if os.path.exists(ModelPath):
        effnetv2s = torch.load(ModelPath)
        return effnetv2s 

    # Setting up the model download path
    path = '/model_files/weights/cnn_model/'
    # Setting download path
    os.environ['TORCH_MODEL_ZOO'] = path 

    # Downloading the model
    effnetv2s = models.efficientnet_v2_s(pretrained=True)

    # Changing the output feature to VocabSize
    NumFeatures = effnetv2s.classifier[1].in_features
    # Changing every parameter to be stable and not dynamic
    for params in effnetv2s.parameters():
        params.requires_grad = False

    effnetv2s.classifier[1] = nn.Linear(in_features=NumFeatures, out_features=VocabSize)
    return effnetv2s

