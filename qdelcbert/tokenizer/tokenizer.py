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
        text: str
            the input text

        Returns
        -------
        list
            the tokenized text
        '''
        return self.tokenizer.tokenize(text)
