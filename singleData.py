import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

class MyData(Dataset):
    def __init__(self, sentences, labels, word2num):
        self.sentences = sentences
        self.labels = labels
        self.word2num = word2num
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        sen = self.sentences[idx]
        sen = [self.word2num[ch] for ch in sen]
        lab = self.labels[idx] - 1
        return sen, lab

def isNecessary(ch):
    if '\u4e00' <= ch <= '\u9fff':
        return True
    elif ch.isalpha():
        return True
    elif ch.isdigit():
        return True
    return False

def getTest(root):
    file = pd.read_csv(root)
    df = pd.DataFrame(file)
    data = []
    for i in range(len(df)):
        d = df[i:i+1]
        data.append(d)
    data = np.array(data)
    x = []
    for sub in data:
        for d in sub:
            d_t = []
            for d in d[0]:
                if isNecessary(d):
                    d_t += d
            d_t = "".join(d_t)
            x.append(d_t)
    return x


def getSenLab(root1, root2):
    file = pd.read_csv(root1)
    df = pd.DataFrame(file)
    data = []
    for i in range(len(df)):
        d = df[i:i + 1]
        data.append(d)
    data = np.array(data)
    sentences = []
    Labels = []
    max_seq_len = 0
    max_seq_num = 0
    for sub in data:
        for d in sub:
            dialogs, labels = d[1], d[2]
            dialogs = dialogs.split('__eou__')
            # dialogs = ["".join(dialog.split()) for dialog in dialogs]
            for i in range(len(dialogs)):
                d_t = []
                for d in dialogs[i]:
                    if isNecessary(ch=d):
                        d_t += d
                dialogs[i] = "".join(d_t)
            # print(dialogs)
            label = []
            while labels > 0:
                label.insert(0, labels % 10)
                labels = labels // 10
            max_seq_num = len(label) if max_seq_num < len(label) else max_seq_num
            for d, l in zip(dialogs, label):
                max_seq_len = len(d) if max_seq_len < len(d) else max_seq_len
                sentences.append(d)
                Labels.append(l)
    file = pd.read_csv(root2)
    df = pd.DataFrame(file)
    data = []
    for i in range(len(df)):
        d = df[i:i+1]
        data.append(d)
    data = np.array(data)
    test_sen = []
    for sub in data:
        for d in sub:
            d_t = []
            for d in d[0]:
                if isNecessary(d):
                    d_t += d
            d_t = "".join(d_t)
            test_sen.append(d_t)
    return sentences, Labels, max_seq_len, max_seq_num, test_sen

def getDict(sentences):
    word2num = dict()
    num2word = dict()
    s = "".join(sentences)
    # 字典自动去重 序号有问题，要重新去重
    s = list(set([ch for ch in s]))
    word2num = {w:i+1 for i, w in enumerate(s)}
    num2word = {i+1:w for i, w in enumerate(s)}
    word2num['_'] = 0
    num2word[0] = '_'
    return word2num, num2word

