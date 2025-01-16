from qdelcbert.tokenizer.tokenizer import Tokenizer
from qdelcbert.pretraining.config import Config
from qdelcbert.pretraining.dataset import MPRCDataset

def test_dataset():
    dataset = MPRCDataset()
    print(dataset[0])

def test_tokenizer():
    tokenizer = Tokenizer()
    dataset = MPRCDataset() 
    datum = dataset[0]
    tokenized = tokenizer.tokenize_pair(datum)
    print(tokenizer.decode(tokenized[0][0]))
    sentences = []
    for i in range(3):
        sentences.append(dataset[i])
    tokenized = tokenizer.tokenize_multiple(sentences)
    print(tokenized)
    


if __name__ == "__main__":
    test_tokenizer()
    # test_dataset()