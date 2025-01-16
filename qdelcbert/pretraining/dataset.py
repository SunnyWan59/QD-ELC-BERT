import torch
import torch.nn as nn
import transformers
from datasets import load_dataset

class MPRCDataset():
    def __init__(self):
        '''class for the mprc data

        Attributes
        ----------

        '''
        self.raw_datasets = load_dataset("glue", "mrpc")
        self.train_dataset = self.raw_datasets['train']
        self.test_dataset = self.raw_datasets['test']

    def __getitem__(self, idx):
        '''returns the item at the given index

        Attributes
        ----------
        idx: int
            the index of the item

        Returns
        -------
        dict
            the item at the given index
        '''
        return (self.train_dataset[idx]['sentence1'], self.train_dataset[idx]['sentence2'])
    

    def __len__(self):
        '''returns the length of the dataset

        Returns
        -------
        int
            the length of the dataset
        '''
        return len(self.train_dataset)
    


    
