import pandas as pd

from singleData import MyData
from singleData import getDict
from singleData import getTest
from singleData import getSenLab
from singleData import my_collate

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = Embedding(vocab_size, input_dim, pad)
        self.positional = PositionalEncoding(input_dim)

        encoder_layer = nn.TransformerEncoderLayer(input_dim, heads_num, hidden_dim, p_drop)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(max_len * input_dim, 6)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.embedding(x) + self.positional()
        x = x.to(torch.float32)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        return self.softmax(x)

class Embedding(nn.Module):
    def __init__(self, vocab_size, input_dim, pad):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=pad)
    def forward(self, X):
        for i in range(len(X)):
            if len(X[i]) < max_len:
                X[i].extend([0] * (max_len - len(X[i])))
        X = torch.LongTensor(X)
        return self.embedding(X)

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim):
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim

    def forward(self):
        pe = np.zeros((max_len, self.input_dim))
        for i in range(max_len):
            for j in range(self.input_dim):
                if j % 2 == 0:
                    pe[i][j] = math.sin(j / pow(10000, 2 * j / self.input_dim))
                else:
                    pe[i][j] = math.cos(j / pow(10000, 2 * j / self.input_dim))
        return torch.from_numpy(pe)

def train_transformer(model, train_data, valid_data):
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = learn_rate)
    batch_number = len(train_data)
    f1 = 0
    for epoch in range(epochs):
        model.train()
        for batch_idx, (X, label) in enumerate(train_data):
            label = Variable(torch.LongTensor(label).to(device))
            # print(label.shape)
            output = model(X)
            # print(output.shape)
            loss = criteon(output, label)
            if (batch_idx+1) % 100 == 0:
                print('epoch', '%04d,' % (epoch+1), 'step', f'{batch_idx+1} / {batch_number}, ', 'loss:', '{:.6f},'.format(loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            confused_matrix = torch.zeros(output_dim, output_dim).to(device)
            for _, (X, label_valid) in enumerate(valid_data):
                x_valid = X
                label_valid = Variable(torch.LongTensor(label_valid).to(device))
                valid_output = model(x_valid)
                # print(valid_output)
                valid_loss = criteon(valid_output, label_valid)
                pred = valid_output.argmax(dim=1)
                # print(pred, label_valid)
                for p, l in zip(pred, label_valid):
                    confused_matrix[p][l] += 1
                total_correct += torch.eq(pred, label_valid).float().sum().item()
                total_num += len(x_valid)
            acc = total_correct / total_num
            acc = total_correct / total_num
            TP, FN, FP = getScore(confused_matrix)
            prec = float(TP / (TP + FP))
            recall = float(TP / (TP + FN))
            F1 = float(2 * prec * recall / (prec + recall))
            print(f'\nValidating at epoch', '%04d'% (epoch+1) , 'acc:{:6f},'.format(acc), 'prec:', '{:.6f},'.format(prec), 'recall:', '{:.6f},'.format(recall), 'F1:{:.6f}'.format(F1))

def getScore(confused_matrix):
    TP, FP, FN = 0, 0, 0
    for i in range(output_dim):
        for j in range(output_dim):
            if i == j:
                TP += confused_matrix[i][j]
            if i >= 1:
                FN += confused_matrix[i][j]
            if j >= 1:
                FP += confused_matrix[i][j]
    return TP, FN - TP, FP - TP

def getTrain():
    root1 = './train_data.csv'
    root2 = './test.csv'
    sentences, labels, max_seq_len, max_seq_num, test_sen = getSenLab(root1, root2)
    word2num, num2word = getDict(sentences + test_sen)
    assert len(word2num) == len(num2word)
    vocab_size = len(word2num)
    dataset = MyData(sentences, labels, word2num)
    train_size = int(len(dataset) * 0.7)
    valid_size = len(dataset) - train_size
    train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size])
    return train_data, valid_data, vocab_size, word2num

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    input_dim = embedding_dim = 256
    pad = 0
    hidden_dim = 150
    output_dim = 6
    learn_rate = 3e-4
    epochs = 10000
    num_layers = 1
    p_drop = 0.4
    max_len = 120
    heads_num = 4

    train_data, valid_data, vocab_size, word2num = getTrain()
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    valid_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=my_collate)

    model = Model()
    model.to(device)
    train_transformer(model, train_data, valid_data)



