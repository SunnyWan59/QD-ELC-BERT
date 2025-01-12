import torch
import torch.nn as nn
import math 
from pretraining.config import Config


class Embedding(nn.Module):
    def __init__(self, config):
        ''' class for the embedding layer

        Attributes
        ----------
        config: Config
            the configuration of the model
        '''
        super(Embedding, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dimension)
        self.position_embeddings = nn.Embedding(config.sequence_length, config.model_dimension)
        self.segment_embedding = nn.Embedding(config.sentence_vocab_size, config.model_dimension)
        self.layer_norm = nn.LayerNorm(config.model_dimension, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, input_ids, segment_ids):
        '''forward pass for the embedding layer

        Attributes
        ----------
        input_ids: torch.Tensor
            the tokenized input IDs
        segment_ids: torch.Tensor
            the segment IDs

        Returns
        -------
        torch.Tensor
            the embedded input
        '''
        token_embedded = self.token_embedding(input_ids)
        position_embedded = self.position_embeddings(torch.arange(input_ids.size(1)).unsqueeze(0).to(input_ids.device))
        segment_embedded = self.segment_embedding(segment_ids)
        embedded = token_embedded + position_embedded + segment_embedded
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        return embedded
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        ''' class for the multi-head attention layer

        Attributes
        ----------
        config: Config
            the configuration of the model
        '''
        super(MultiHeadAttention, self).__init__()
        self.config = config
        self.linear = nn.Linear(config.model_dimension, config.model_dimension)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x, mask=None):
        '''forward pass for the multi-head attention layer

        Attributes
        ----------
        x: torch.Tensor
            the input tensor
        mask: torch.Tensor default = None
            the mask tensor

        Returns
        -------
        torch.Tensor
            the output tensor
        '''
        q = self.linear(x)
        k = self.linear(x)
        v = self.linear(x)
        head_dim = self.config.model_dimension // self.config.num_attention_heads

        # reshaping the tensors q,r,v
        q = q.view(x.size(0), x.size(1), self.config.num_attention_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(x.size(0), x.size(1), self.config.num_attention_heads, head_dim).permute(0, 2, 1, 3) #change this
        v = v.view(x.size(0), x.size(1), self.config.num_attention_heads, head_dim).permute(0, 2, 1, 3)

        #scaled dot product attention
        scores = nn.functional.scaled_dot_product_attention(q, k, v, mask)

        attention_probs = nn.Softmax(dim=-1)(scores)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(1), self.config.model_dimension)
        output = self.out(context)
        return output


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
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

class SegmentationEmbedding(nn.Module):
    def __init__(self,
                 input_ids,
                 sep_token_id = 102
                 ):
        '''class for segmentation embedding

        Attributes
        ----------
        input_ids: torch.Tensor 
            tokenized input IDs 
        sep_tocken_id: int default = 102
            token ID for the [SEP] token
        '''
        super(SegmentationEmbedding, self).__init__()
        self.input_ids = input_ids
        self.sep_token_id = sep_token_id

    def _create_segmentation_embedding(input_ids, sep_token_id=102):
        segment_embedding = torch.zeros_like(input_ids)
        segment_id = 0
        for i, token_id in enumerate(input_ids):
            segment_embedding[i] = segment_id
            if token_id == sep_token_id:
                segment_id = 1
        return segment_embedding
    
    def forward(self,x):
        return x + self._create_segmentation_embedding(self.input_ids, self.sep_token_id)

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


