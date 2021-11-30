import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

def getDict(sentences):
    sentences = " ".join(sentences)
    s = list(set(sentences.split()))
    word2num = {w:i for i, w in enumerate(s)}
    num2word = {i:w for i, w in enumerate(s)}

    return word2num, num2word

class Embedding(nn.Module):
    def __init__(self, vocab_size, input_dim, pad):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=pad)
        # torch.nn.Embedding(num_embeddings, embedding_dim,  词典数量，维度
        # padding_idx=None, max_norm=None, norm_type=2.0, 补零的位置, 超过范数归一化，指定范数
        # scale_grad_by_freq=False, sparse=False, _weight=None, 梯度放缩，系数张量
        # device=None, dtype=None)
    def forward(self, X):
        # X : [batch_size, seq_len, embedding_dim]
        return self.embedding(X)

if __name__ == '__main__':
    sentences = ["i like milk", "i like meat", "i hate dog", "i hate cat", "hello world"]
    word2num, num2word = getDict(sentences)
    vocab_size = len(word2num)
    seq_len = 3
    x = []
    for sen in sentences:
        ss = []
        for s in sen.split():
            ss.append(int(word2num[s]))
        while len(ss) < seq_len:
            ss.append(0)
        x.append(ss)
    x = Variable(torch.LongTensor(x))
    model = Embedding(vocab_size, input_dim=10, pad=0)
    y = model(x)
    print(x, y, y.shape, sep = '\n---\n')

