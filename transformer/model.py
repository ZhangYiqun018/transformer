import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = Embedding(vocab_size, input_dim, pad)
        self.pe = PositionalEncoding(input_dim)

        encoder_layer = nn.TransformerEncoderLayer(input_dim, heads_num, hidden_dim, p_drop)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(input_dim, heads_num, hidden_dim, p_drop)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('inf').masked_fill(mask==1, float(0, 0)))
        return mask

    def forward(self, src, tgt):
        x = (self.embedding(src) + self.pe()).to(torch.float32)
        y = (self.embedding(tgt) + self.pe()).to(torch.float32)
        m = self.encoder(x)
        output = self.decoder(y, m)
        return output

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


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_layers = 1
    vocab_size = 100
    input_dim = 256
    max_len = 128
    pad = 0
    batch_size = 32
    heads_num = 8
    hidden_dim = 128
    p_drop = 0.1


    model = Model().to(device)
    print(model)

    x = [[i for i in range(50)] for _ in range(32)]
    y = x
    print(len(x), len(x[0]))
    # x = torch.tensor([32, 32, 256])
    out = model(x, y)

    print(out.shape)
