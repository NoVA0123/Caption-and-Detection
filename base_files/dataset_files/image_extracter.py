import torch
from torchvision.transforms import v2
from torchvision.io import read_image
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Class for dataset loader
class imgextracter(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        # Image transformation
        self.transform = v2.Compose([
            v2.ToDtype(torch.float, scale=True),
            v2.Resize(size=[224,224]),
            v2.Normalize(mean=[0,0,0], std=[1,1,1])
            ])
        self.transform = self.transform.to(device)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe['image_path'][index] # Path of the image
        img = read_image(row).to(device)
        return self.transform(img) # Transform the image
