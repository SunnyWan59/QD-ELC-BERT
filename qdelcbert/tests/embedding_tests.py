import torch
import torch.nn as nn
import sys
import os 

from qdelcbert.tokenizer.tokenizer import Tokenizer
from qdelcbert.pretraining.config import Config
from qdelcbert.model.model import Embedding, MultiHeadAttention, _SegmentationEmbedding


def test_tokenizer():
    tokenizer = Tokenizer()
    assert tokenizer.tokenizer is not None
    text = "Using a Transformer network is simple"
    ids = tokenizer.tokenize(text)
    print(ids)
    print(tokenizer.decode(ids[0][0]))

    text2 = '[CLS][SEP]Hi[SEP]'
    ids2 = tokenizer.tokenize(text2)
    print(ids2)


def test_embedding():
    config = Config(
        model_dimension=4,
        num_attention_heads=2,
    )
    embedding = Embedding(config)
    print(embedding)

    tokenizer = Tokenizer()
    text = ["Using a Transformer network is simple"]
    input = tokenizer.tokenize(text)[0]
    # segment_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
    output = embedding(input)
    print(output)


def test_seg_embedding():
    config = Config(
        model_dimension=4,
        num_attention_heads=2,
    )
    seg_embedding = _SegmentationEmbedding(config)
    tokenizer = Tokenizer()
    text = ["Using a Transformer network is simple"]
    input = tokenizer.tokenize(text)[0]
    print (input)
    print(seg_embedding._create_segmentation_mask(input))

if __name__ == "__main__":
    test_tokenizer()
    # test_embedding()
    # test_seg_embedding()