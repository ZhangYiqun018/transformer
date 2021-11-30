import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.Layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, X):
        return self.Layer(X)

def getDict(sentences):
    sentences = " ".join(sentences)
    s = list(set(sentences.split()))
    word2num = {w:i for i, w in enumerate(s)}
    num2word = {i:w for i, w in enumerate(s)}

    return word2num, num2word

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn([1, 40]).to(device)
    model = FeedForward(40, 100)
    model.to(device)
    output = model(x)



