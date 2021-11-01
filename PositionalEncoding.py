import math
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim):
        super(PositionalEncoding, self).__init__()
        self.input_dim = input_dim

    def forward(self, X):
        # X : [batch_size, seq_len, input_dim]
        seq_len = X.shape[1]
        pe = np.zeros((seq_len, self.input_dim))
        for i in range(seq_len):
            for j in range(self.input_dim):
                if j % 2 == 0:
                    pe[i][j] = math.sin(j / pow(10000, 2*j / self.input_dim))
                else:
                    pe[i][j] = math.cos(j / pow(10000, 2*j / self.input_dim))
        print(pe)
        return torch.from_numpy(pe)


if __name__ == '__main__':
    # x : [batch_size, sep_len, input_dim]
    X = torch.randn([3, 10, 20])

    model = PositionalEncoding(20)
    y = model(X)
    z = X + y
    print(y, y.shape, z, z.shape)