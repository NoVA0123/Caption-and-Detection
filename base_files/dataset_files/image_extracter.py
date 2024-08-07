import torch
from torchvision.transforms import v2
from torchvision.io import read_image
import pandas as pd


# Class for dataset loader
class imgextracter(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe['image_path'][index] # Path of the image
        return read_image(row) # Transform the image
