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
    print(tokenizer.tokenize("Using a Transformer network is simple"))

def test_embedding():
    config = Config(
        vocab_size=6,
        model_dimension=4,
        num_attention_heads=2,
    )
    embedding = Embedding(config)
    print(embedding)

    tokenizer = Tokenizer()
    tokenizer.tokenize("Using a Transformer network is simple")
    
    


    # input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    # segment_ids = torch.tensor([[0, 0, 0, 0, 0]])
    # output = embedding(input_ids, segment_ids)
    # assert output.size() == (1, 5, 768)
    # print("Embedding test passed")


if __name__ == "__main__":
    test_tokenizer()
    test_embedding()