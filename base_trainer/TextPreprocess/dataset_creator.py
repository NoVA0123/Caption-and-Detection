import re
import json
import pandas as pd


# Function to extract Image Path and caption path
def extracter(JsonPath: str,
              ImgPath: str) -> pd.DataFrame:
    # Loading Json
    with open(JsonPath, 'r') as f:
        data = json.load(f)
        data = data['annotations']

    # Creating main list for details of the image
    ImgCap = []
    
    # Traversing the json
    for sample in data:
        '''Image file name is stored as 12 digits + .jpg and in annoations the
           image is stored as id and it is not of 12 digits. So, we need to
           to 12 digit string
        '''
        ImgName = '%012d.jpg' % sample['image_id'] # twelve 0's and .jpg
        ImgName = f"{ImgPath}{ImgName}" # add digits from right
        ImgCap.append([ImgName, sample['caption']])

    # Creating a Data Frame for preprocessing the text
    captions = pd.DataFrame(ImgCap, columns=['imgae_path', 'caption'])
    return captions


# Creating dataset from annotations
def create_dataset(TrainAnn: str,
                   ValAnn: str,
                   TrainImg: str,
                   ValImg: str) -> tuple:

   # Training Data
   TrainData = extracter(TrainAnn, TrainImg)

   # Training Data
   ValData = extracter(ValAnn, ValImg)

   return TrainData, ValData


# Function for preprocessing text
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Removes alpha numeric and unicode white spaces like '\t' and '\n'
    text = '[SOS]' + text + '[EOS]' # Adding start and end for embedding process
    return text


# Length finder
def length_finder(TrainData: pd.DataFrame,
                  ValData: pd.DataFrame) -> int:
    MaxLen = 0

    for i in TrainData['caption'].tolist():
        m = i.split()
        if len(m) > MaxLen:
            MaxLen = len(i)

    for i in ValData['caption'].tolist():
        m = i.split()
        if len(m) > MaxLen:
            MaxLen = len(i)

    return MaxLen
