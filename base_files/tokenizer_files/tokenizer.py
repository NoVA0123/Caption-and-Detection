import os
from tokenizers import (decoders,
                        models,
                        normalizers,
                        pre_tokenizers,
                        processors,
                        trainers,
                        Tokenizer)
from transformers import PreTrainedTokenizerFast
import pandas as pd
import torch
from tqdm.auto import tqdm


# Function to build tokenizer
def get_tokenizer(dataset:pd.DataFrame,
                  path:str='tokenizer.json') -> Tokenizer:
    if os.path.exists(path):
        tokenizer = Tokenizer.from_file(path)
        return tokenizer


    tokenizer = Tokenizer(models.WordPiece(unk_token='[UNK]')) # To represent Unkown words
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # Words are seperated by spaces
    
    ''' Start and end of a sentence should be defined and also the minimum appearance
    of a word should be 2.
    '''

    trainer = trainers.WordPieceTrainer(special_tokens=["[UNK]",
                                                        "[PAD]",
                                                        "[SOS]",
                                                        "[EOS]"],
                                        min_frequency=2)

    # Training the tokenizer
    tokenizer.train_from_iterator(get_all_sentences(dataset),
                                  trainer=trainer)

    # Changing the decoder
    tokenizer.decode = decoders.WordPiece()

    # Changing Post processor
    SosToken = tokenizer.token_to_id('[SOS]')
    EosToken = tokenizer.token_to_id('[EOS]')
    tokenizer.post_processor = processors.TemplateProcessing(
            single=f'[SOS]:0 $A:0 [EOS]:0',
            special_tokens=[('[SOS]', SosToken), ('[EOS]', EosToken)])

    # Saving the tokenizer
    tokenizer.save(path)
    
    return tokenizer


def fast_tokenizer(tokenizer:Tokenizer,
                   MaxSeqLen:int) -> PreTrainedTokenizerFast:

    return PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                   unk_token='[UNK]',
                                   pad_token='[PAD]',
                                   bos_token='[SOS]', # Begining of sentence
                                   eos_token='[EOS]',
                                   padding_side='right',
                                   model_max_length=MaxSeqLen)


def get_all_sentences(dataset: pd.DataFrame):
    
    for item in dataset['caption'].tolist():
        yield item



# Convert text to id class
class texttoid:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerFast,
                 dataset: pd.DataFrame):

        self.dataset = dataset
        self.tokenizer = tokenizer # Tokenized Tensor
        self.padToken = tokenizer.convert_tokens_to_ids('[PAD]')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> dict:

        row = self.dataset['caption'][index]
        DecoderInput = self.tokenizer(text=row,
                                      padding='max_length',
                                      return_tensors='pt') # Tokenized sentence
        DecoderInput = DecoderInput['input_ids']

        # Label should 1 value ahead of input
        Label = torch.cat([
            DecoderInput[1:],
            torch.tensor([self.padToken])
            ])
        
        return{
                "decoder_input": DecoderInput,
                "label": Label
                }

