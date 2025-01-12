import torch 
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self,
                 config, 
                 vocab_size, embedding_size):
        ''' class for the embedding layer

        Attriubutes
        ----------
        vocab_size: int
            the size of the vocabulary
        embedding_size: int
            should be d_model, the same dimension as the positional encoder
        '''
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size) # for now I will use pytorch's embedding layer
        
    def forward(self, x):
        return self.embedding(x)
    
class PositionalEncoder(nn.Module):
    def __init__(self, 
                model_dimension,
                sequence_length,
                frequency_scalar = 10000
                ):
        ''' class for the positonal encoder

        Attributes
        ---------
        model_dimension: int
            the dimension of the model, should be the same as the embedding size 
        sequence_lenth: int
            the length of the sequence 
        frequency_scalar: int default = 10000
            arbitrary frequency scalar

        '''
        super(PositionalEncoder, self).__init__()
        self.model_dimension = model_dimension
        self.sequence_len = sequence_length
        self.n = frequency_scalar

    def _get_positional_encoding(self, sequence_len, model_dimension,n):
        positional_encoding = torch.zeros((sequence_len, model_dimension), dtype=torch.float32)
        for pos in range(sequence_len):
            for i in range(0, model_dimension, 2):
                positional_encoding[pos, i] = math.sin(pos / (n ** ((2 * i)/model_dimension)))
                positional_encoding[pos, i + 1] = math.cos(pos / (n ** ((2 * (i + 1))/model_dimension)))
        return positional_encoding

    def forward(self, x):
        return x + self._get_positional_encoding(self.sequence_len, self.model_dimension, self.n) # adding the positional encoding to the input

