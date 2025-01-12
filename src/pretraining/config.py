import torch 
import torch.nn as nn

class Config():
    def __init__(self,
                 vocab_size,
                 model_dimension,
                 num_attention_heads,
                 sentence_vocab_size = 2,
                 sequence_length = 512,
                 dropout_prob = 0.1
                 ):
        '''class for the configuration of the model
        '''
        self.vocab_size = vocab_size
        self.model_dimension = model_dimension
        self.num_attention_heads = num_attention_heads
        self.sentence_vocab_size = sentence_vocab_size
        self.sequence_length = sequence_length