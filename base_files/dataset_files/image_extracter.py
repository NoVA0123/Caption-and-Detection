import torch
from torchvision.transforms import v2
from torchvision.io import read_image
import pandas as pd


# Class for dataset loader
class imgextracter(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.transform = v2.Compose([
            v2.ToDtype(torch.float, scale=True), # Scale the image
            v2.Resize(size=[224, 224]), # Resisze for the model
            v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), # Normalize values
            v2.ToDtype(torch.float) # change it back to float
            ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe['image_path'][index] # Path of the image
        img = read_image(row) # Read the image located on the path
        return self.transform(img) # Transform the image
