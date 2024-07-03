import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: %(message)s:')


# List of files
list_of_files = [
        'src/__init__.py',
        'src/helper.py',
        'src/prompt.py'
        '.env',
        'base/self_attention.py',
        'base/multi_head_attention.py',
        'base/decoder.py',
        'base/embedding.py',
        'base/encoder.py',
        'base/feed_forward.py',
        'base/layer_normalization.py',
        'base/positional_encoding',
        'base/transformer.py',
        'base/tokenizer.py',
        'model_files/tokenizer/',
        'model_files/weights/',
        'requirements.txt',
        ]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory; {filedir} for the file {filename}')


    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)) == 0:
        with open(filepath, 'w') as f:
            pass
        logging.info(f'Creating empty file: {filepath}')

    else:
        logging.info(f'{filename} is already created')
