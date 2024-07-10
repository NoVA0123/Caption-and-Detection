from pathlib import Path

def get_config(MaxSeqLen: int,
               Epoch: int=5):
    return {
            'batch_size': 8,
            'num_epochs': Epoch,
            'lr': 10**-4,
            'seq_len': MaxSeqLen,
            'd_model': 512,
            'model_folder': 'model_files/weights/caption_model',
            'model_basename': 'caption_model',
            'preload': None,
            'tokenizer_file': 'tokenizer.json',
            'experiment_name': 'runs/tmodel'
            }


def get_weights_file_path(config, epoch: str):
    ModelFolder = config['model_folder']
    ModelBasename = config['model_basename']
    ModelFilename = f'{ModelBasename}{epoch}.pt'
    return str(Path('.') / ModelFolder / ModelFilename)
