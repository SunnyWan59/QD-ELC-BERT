from qdelcbert.tokenizer.tokenizer import Tokenizer
from qdelcbert.pretraining.config import Config
from qdelcbert.pretraining.dataset import MPRCDataset

def test_dataset():
    dataset = MPRCDataset()
    print(dataset[0])

if __name__ == "__main__":
    test_dataset()