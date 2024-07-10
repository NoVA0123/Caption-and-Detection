from os.path import exists
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
import torch
from tqdm.auto import tqdm


# Function to create tokenizer
def tokenizer_creator(Dataset: pd.DataFrame, path: str) -> Tokenizer:
    # If tokenizer already exists, import it.
    if exists(path):
        tokenizer = Tokenizer.from_file(path)
        return tokenizer
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # To fix 'unkown word' problem
    tokenizer.pre_tokenizer = Whitespace() # Delimiter for words is set to space using this function
    # Start, End, Unkown, Padding and minimum appearance of the word.
    trainer = WordLevelTrainer(special_tokens=['[UNK]',
                                               '[PAD]',
                                               '[SOS]',
                                               '[EOS]'], min_frequency=2)

    # Training the tokenizer to make its own vocabulary
    tokenizer.train_from_iterator(
            GetAllSentences(Dataset['caption'].tolist()),
            trainer=trainer
            )
    # Saving the tokenizer
    tokenizer.save(path)
    return tokenizer


# Function to yield all sentences
def GetAllSentences(sentenceList:list):
    for item in sentenceList:
        yield item


# Class object to convert sentences into tokens using tokenizer
class texttoid:
    def __init__(self,
                 tokenizer: Tokenizer,
                 MaxSeqLen: int) -> None:

        self.tokenizer = tokenizer
        self.seqLen = MaxSeqLen # Max length of a sentence in dataset

        # Initialize Start, End and Padding tokens
        self.sosToken = torch.tensor(
                [tokenizer.token_to_id('[SOS]')],
                dtype=torch.uint8
                )
        self.eosToken = torch.tensor(
                [tokenizer.token_to_id('[EOS]')],
                dtype=torch.uint8
                )
        self.padToken = torch.tensor(
                [tokenizer.token_to_id('[PAD]')],
                dtype=torch.uint8
                )


    def converter(self, sentence:str) -> dict:
        # Encodes the sentences
        DecodeInputTok = self.tokenizer.encode(sentence).ids
        # Number of Padded tokens in a sentence
        DecodeNumPadTok = self.seqLen - len(DecodeInputTok) - 1
        
        # Concatenating Start, Decoder sentence and Padded Tokens
        DecoderInput = torch.cat(
                [
                    self.sosToken,
                    torch.tensor(DecodeInputTok, dtype=torch.short),
                    torch.tensor([self.padToken] * DecodeNumPadTok, dtype=torch.uint8),
                    ]
                )
        
        # Label will be next word after current word
        # Concatenating Decoder sentence, End and Padded Tokens to form label
        Label = torch.cat(
                [
                    torch.tensor(DecodeInputTok, dtype=torch.short),
                    self.eosToken,
                    torch.tensor([self.padToken] * DecodeNumPadTok, dtype=torch.uint8),
                    ]
                )
        

        return {
                "decoder_input": DecoderInput,
                "label": Label
                }
    pass


# Function to convert the sentences into tokens
def convert(Dataset: pd.DataFrame,
            tokenizer: Tokenizer,
            MaxLen: int) -> list:
    # Assign the converter object
    Converter = texttoid(tokenizer,
                         MaxLen)
    data = [] # A list that will contain tokenized sentences
    # Loop to append tokenized sentences
    for x in tqdm(Dataset['caption'].tolist(), ascii=True, desc='Converting captions'):
        data.append(Converter.converter(x))
    return data
