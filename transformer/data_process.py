import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self, de, en, word2num):
        self.de = de
        self.en = en
        self.word2num = word2num
    def __len__(self):
        return len(self.de)
    def __getitem__(self, idx):
        en = self.en[idx]
        de = self.de[idx]
        en = [self.word2num[en] for ch in en]
        de = [self.word2num[de] for ch in de]
        return de, en

def getDict():
    root = "./data/bpevocab"
    f = open(root, 'rt')
    data = f.readlines()
    vocab_size = len(data)

    word2num = {}
    num2word = {}
    for d in data:
        d = d.split()
        word2num[d[0]] = d[1]
        num2word[d[1]] = d[0]
    return word2num, num2word

if __name__ == '__main__':
    word2num, num2word = getDict()
    vocab_size = len(word2num)




