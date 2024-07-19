import os
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
import torch
from tqdm.auto import tqdm


# Function to build tokenizer
def get_tokenizer(dataset:pd.DataFrame,
                  path:str='tokenizer.json') -> Tokenizer:
    if os.path.exists(path):
        tokenizer = Tokenizer.from_file(path)
        return tokenizer


    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # To represent Unkown words
    tokenizer.pre_tokenizer = Whitespace() # Words are seperated by spaces
    
    '''
    Start and end of a sentence should be defined and also the minimum appearance
    of a word should be 2.
    '''

    trainer = WordLevelTrainer(special_tokens=["[UNK]",
                                               "[PAD]",
                                               "[SOS]",
                                               "[EOS]"],
                               min_frequency=2)

    tokenizer.train_from_iterator(get_all_sentences(dataset),
                                  trainer=trainer)
    tokenizer.save(path)
    
    return tokenizer


def get_all_sentences(dataset: pd.DataFrame):
    
    for item in dataset['caption'].tolist():
        yield item



# Convert text to id class
class texttoid:
    def __init__(self,
                 tokenizer: Tokenizer,
                 MaxSeqLen: int,
                 dataframe: pd.DataFrame):

        self.tokenizer = tokenizer
        self.maxSeqLen = MaxSeqLen
        self.dataframe = dataframe

        self.sosToken = torch.tensor([tokenizer.token_to_id('[SOS]')],
                                     dtype=torch.int)
        self.eosToken = torch.tensor([tokenizer.token_to_id('[EOS]')],
                                     dtype=torch.int)
        self.padToken = torch.tensor([tokenizer.token_to_id('[PAD]')],
                                     dtype=torch.int)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> dict:
        
        x = self.dataframe['caption'][index] # Caption

        # Turning sentences into respective token ids
        DecInputTok = self.tokenizer.encode(x).ids
        
        # Finding number of Padding tokens
        NumPadTok = self.maxSeqLen - len(DecInputTok)

        # Concatenating tensors (Sos, Input, Padding)
        DecoderInput = torch.cat([
            self.sosToken,
            torch.tensor(DecInputTok, dtype=torch.int),
            torch.tensor([self.padToken] * NumPadTok, dtype=torch.int)
            ])

        label = torch.cat([
            torch.tensor(DecInputTok, dtype=torch.int),
            self.eosToken,
            torch.tensor([self.padToken] * NumPadTok, dtype=torch.int)
            ])


        return{
                "decoder_input": DecoderInput,
                "label": label
                }

