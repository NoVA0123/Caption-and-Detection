# Caption


Requirements:

```
pip install -r requirements.txt
```

- Python           3.9.13
- Pytorch
- Triton           3.0.0
- HuggingFace Hub
- Numpy            1.26.4
- Pillow



Note: This is not a proper working model because it is trained on 128 steps rather than 20k - 90k steps. "validation_output.txt" have results of training. Each line displays an output for each step.


# Model Description


This Model uses pretrained EfficientNet5 for feature extraction and GPT-2 architecture (from scratch) is used as decoder.



Model Specification:

- Sequence Length = 1024
- Token Dimension(dmodel) = 768
- Image size (after transformation) = 489 x 456
- Number of Decoder Layers = 12
- Num of Decoder Heads = 12
- Batch Size = 4
- Optimzer = AdamW


This model uses decay rate and grad clipping through normalization to reduce shock while learning.


Features used to speed up training:

- Distributed Data Parallel
- Gradient Clipping
- Torch Compile
- Fused Adam
- FP16

