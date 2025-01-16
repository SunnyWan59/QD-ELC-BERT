from qdelcbert.tokenizer.tokenizer import Tokenizer
from qdelcbert.pretraining.config import Config
from qdelcbert.pretraining.dataset import MPRCDataset

def test_dataset():
    dataset = MPRCDataset()
    print(dataset[0])

def test_tokenizer():
    tokenizer = Tokenizer()
    dataset = MPRCDataset() 
    datum = [dataset[0]]
    # print(datum[0])
    # print(tokenizer.tokenize(datum[0]))
    # print(datum)
    tokenized = tokenizer.tokenize_pair(datum[0])
    print(tokenized)
    print(tokenizer.decode(tokenized[0][0]))
if __name__ == "__main__":
    test_tokenizer()
    # test_dataset()