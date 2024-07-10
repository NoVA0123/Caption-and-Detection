from config import get_config
from model import build_transformer
from tokenizers import Tokenizer
import torch
from base_trainer.vision_model.vision_model import vision_model
from argparse import ArgumentParser
from torchvision.transforms import v2
from torchvision.io import read_image
import warnings


def translate(ImgPath: str,
              TokenizerPath: str,
              ModelPath:str,
              CnnModelPath=None,
              SpecifiedPath=None):

    # Storing device identity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    # Initializing model
    MaxLen = 75
    effnetv2s = vision_model(MaxLen, CnnModelPath, SpecifiedPath)
    config = get_config(MaxLen)
    tokenizer = Tokenizer.from_file(TokenizerPath)
    model = build_transformer(effnetv2s, tokenizer.get_vocab_size(), MaxLen)

    # Loading the model
    ModelFilename = ModelPath
    state = torch.load(ModelFilename)
    model.load_state_dict(state['model_state_dict'])


    # translate
    model.eval()
    with torch.no_grad():

        # Read the image
        transform = v2.Compose([
            v2.ToDtype(torch.float, scale=True), # turns images into float16
            v2.Resize(size=[224, 224]), # Resizes the image
            v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), # Normalize values
            v2.ToDtype(torch.float)
            ])

        Img = read_image(ImgPath)
        Img = transform(Img)
        Img = Img.unsqueeze(0)

        # Encoding the image
        EncoderOutput = model.encode(Img, MaxLen, 1).to(device)

        # Blueprint for Decoder Input
        TmpId = 34
        tmp = torch.cat([
            torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.long),
            torch.zeros(TmpId, dtype=torch.long),
            torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.long),
            torch.tensor([tokenizer.token_to_id('[PAD]')] * (MaxLen - TmpId - 2),
                         dtype=torch.long),
            ], dim=0)
        # Initializing the decoder
        DecoderInput = torch.empty(1,
                                   1).fill_(
                                           tokenizer.token_to_id('[SOS]')
                                           ).type_as(tmp).to(device)


        # Generating Caption
        while DecoderInput.size(1) < MaxLen:
            Output = model.decode(DecoderInput, EncoderOutput)

            # Project next toekn
            prob = model.project(Output[:, -1])
            _, NextWord = torch.max(prob, dim=1)
            DecoderInput = torch.cat([DecoderInput,
                                       torch.empty(1, 1).type_as(tmp).fill_(NextWord.item()).to(device)],
                                      dim=1)

            # Translate the sentence
            print(f"{tokenizer.decode([NextWord.item()])}", end=' ')


            # Break if we predicted the sentece
            if NextWord == tokenizer.token_to_id("[EOS]"):
                break


    # Convert ids to tokens
    return tokenizer.decode(DecoderInput[0].tolist())


def get_command_line_arguments():
    parser = ArgumentParser()
    # Specifies required paths
    parser.add_argument("--paths",
                        dest='Paths',
                        required=True,
                        action='append',
                        help='Specify the paths')
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = get_command_line_arguments()
    args = args.Paths
    translate(*args)
