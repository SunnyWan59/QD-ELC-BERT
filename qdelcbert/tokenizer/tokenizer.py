import torch 
from torch import nn
from transformers import BertTokenizer

'''
Implemenntation of WordPiece Tokenizer used in BERT
Will be completed last
'''

class Tokenizer():
    def __init__(self):
        ''' class for the tokenizer

        Attributes
        ----------
        tokenizer: BertTokenizer
            the tokenizer
        '''
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize(self, text):
        '''tokenizes the input text

        Attributes
        ----------
        text: [str]
            the input text

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            the tokenized text, and the attention mask 
        '''
        encoding = self.tokenizer.batch_encode_plus(
            text,                	
            padding=True,          	
            truncation=True,       	
            return_tensors='pt',  	
            add_special_tokens=True
        )   

        return((encoding['input_ids'], encoding['attention_mask']))
    
    def decode(self, token_ids):
        '''decodes the tokenized text

        Attributes
        ----------
        token_ids: torch.Tensor
            the tokenized text

        Returns
        -------
        str
            the decoded text
        '''
        return self.tokenizer.decode(token_ids)
