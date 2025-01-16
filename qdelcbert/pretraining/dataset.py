import torch
import torch.nn as nn
import transformers
from datasets import load_dataset

class MPRCDataset():
    def __init__(self):
        '''class for the mprc data

        Attributes
        ----------
        raw_datasets : dict
            the raw datasets
        train_dataset : dict
            the training dataset
        test_dataset : dict
            the test dataset
        '''
        self.raw_datasets = load_dataset("glue", "mrpc")
        self.train_dataset = self.raw_datasets['train']
        self.test_dataset = self.raw_datasets['test']
        # This is SUPER inefficient, but it's just for now.
    
    

    def __len__(self):
        '''returns the length of the dataset

        Returns
        -------
        int
            the length of the dataset
        '''
        return len(self.train_dataset)
    


    
