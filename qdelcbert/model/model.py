import torch
import torch.nn as nn
import math 
from qdelcbert.pretraining.config import Config



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
        self.position_embeddings = AbsolutePositionEmbedding(config)
        self.segment_embedding = nn.Embedding(config.sentence_vocab_size, config.model_dimension)
        self.layer_norm = nn.LayerNorm(config.model_dimension, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def __str__(self):
        return (
            "self.token_embedding: " + str(self.token_embedding) + "\n" +
            "self.position_embeddings: " + str(self.position_embeddings) + "\n"
            "self.segment_embedding: " + str(self.segment_embedding) + "\n"
                )

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
        print(torch.arange(input_ids.size(0)).unsqueeze(0).to(input_ids.device))
        position_embedded = self.position_embeddings(input_ids)
        segment_embedded = self.segment_embedding(segment_ids)
        # print(token_embedded.size())
        # print(position_embedded.size())
        # print(segment_embedded.size())
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
        k = k.view(x.size(0), x.size(1), self.config.num_attention_heads, head_dim).permute(0, 2, 1, 3) 
        v = v.view(x.size(0), x.size(1), self.config.num_attention_heads, head_dim).permute(0, 2, 1, 3)

        #scaled dot product attention
        scores = nn.functional.scaled_dot_product_attention(q, k, v, mask)

        attention_probs = nn.Softmax(dim=-1)(scores)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(1), self.config.model_dimension)
        output = self.out(context)
        return output
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        ''' class for the feed forward mlp layer
        '''
        super(FeedForwardNetwork, self).__init__()
        self.config = config
        


class _SegmentationEmbedding(nn.Module):
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
        super(_SegmentationEmbedding, self).__init__()
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
    

class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, config):
        super(AbsolutePositionEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(config.sequence_length , config.model_dimension)
        #ininitilize the position embeddings with random weights
        nn.init.normal_(self.position_embeddings.weight, mean=0, std=0.02)
    
    def forward(self, input_ids):
        '''
        
        Parameters
        ----------
        input_ids : torch.Tensor
            the tokenized input IDs
        '''
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # Shape: (batch_size, seq_length)
        
        # Lookup position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings