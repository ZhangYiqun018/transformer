import math
import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, dim_q, dim_k, dim_v, heads_num, required_mask=False):
        super(MultiHeadAttention, self).__init__()
        assert dim_k % heads_num == 0
        assert dim_v % heads_num == 0
        self.Q = nn.Linear(input_dim, dim_q)
        self.K = nn.Linear(input_dim, dim_k)
        self.V = nn.Linear(input_dim, dim_v)
        self.out = nn.Linear(dim_v, input_dim)

        self.heads_num = heads_num
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.required_mask = required_mask
    def getMask(self, dim):
        mask = np.ones((dim, dim))
        mask = torch.tensor(np.tril(mask))
        return mask.bool()
    def forward(self, X):
        # X                         : [batch_size, seq_len, input_dim]
        # Q_pre                     : [batch_size, seq_len, dim_q]
        # Q                         : [_, batch_size, seq_len, dim_q // heads]
        # K.permute(0, 1, 3, 2)     : [_, batch_size, dim_q // heads, seq_lem]
        # output                    : [_, batch_size, seq_len, seq_len]
        # mask                      : [seq_len, seq_len]
        Q = self.Q(X).reshape(-1, X.shape[0], X.shape[1], self.dim_q // self.heads_num)
        K = self.K(X).reshape(-1, X.shape[0], X.shape[1], self.dim_k // self.heads_num)
        V = self.V(X).reshape(-1, X.shape[0], X.shape[1], self.dim_v // self.heads_num)
        output = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.dim_k)
        if self.required_mask == True:
            mask = self.getMask(X.shape[1])
            print(output.shape, mask.shape)
            output = torch.masked_fill(output, mask, value=float("-inf"))
        output = nn.Softmax(-1)(output)
        output = torch.matmul(output, V).reshape(X.shape[0], X.shape[1], -1)
        return self.out(output)

class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.Q = nn.Linear(input_dim, dim_q)
        self.K = nn.Linear(input_dim, dim_k)
        self.V = nn.Linear(input_dim, dim_v)
        self.dim_k = dim_k
    def forward(self, X):
        q = self.Q(X)
        k = self.K(X)
        v = self.V(X)
        output = torch.matmul(q, k.permute(0, 2, 1))
        output = torch.matmul(nn.Softmax(dim=-1)(output / math.sqrt(self.dim_k)), v)
        return output

if __name__ == '__main__':
    """
        X : [batch_size, seq_len, embedding_dim(input_dim)]
        input_dim : embedding_dim
        dim_q : 
        dim_k : 与dim_q 相同
        dim_v : 
        output : [batch_size, seq_len, dim_v]
    """
    X = torch.randn([4, 3, 10])
    print(X, X.type())
    # model = SelfAttention()
    model1 = SelfAttention(10, 4, 4, 8)
    model2 = MultiHeadAttention(input_dim=10, dim_q=20, dim_k=20, dim_v=8, heads_num=2, required_mask=True)
    y1 = model1(X)
    y2 = model2(X)
    # print(y1, y1.shape)
    # print(y2, y2.shape)
    print(y1.shape, y2.shape)