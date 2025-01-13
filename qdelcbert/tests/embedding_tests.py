import torch
import torch.nn as nn
import sys
import os 

from qdelcbert.tokenizer.tokenizer import Tokenizer
from qdelcbert.pretraining.config import Config
from qdelcbert.model.model import Embedding, MultiHeadAttention


def test_tokenizer():
    tokenizer = Tokenizer()
    assert tokenizer.tokenizer is not None
    text = ["Using a Transformer network is simple", "I love transformers!"]
    ids = tokenizer.tokenize(text)
    print(ids)
    # print(ids[0].size(1))
    print(tokenizer.decode(ids[0][0]))
    # comb_ids = torch.cat((ids[0][0],ids[0][1][1:]))
    # print(comb_ids)
    # print(tokenizer.decode(comb_ids))

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
    segment_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
    output = embedding(input, segment_ids)
    print(output)


if __name__ == "__main__":
    # test_tokenizer()
    test_embedding()