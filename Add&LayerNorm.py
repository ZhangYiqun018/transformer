import torch
import torch.nn as nn

class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
    def forward(self, X):
        print(X.shape[1:], X.size(), X.size()[1:])
        ln = nn.LayerNorm(X.size()[1:])
        return ln(X)

if __name__ == '__main__':
    model = Norm()
    x = torch.randn([4, 3, 10])
    y = model(x)
    print(y, y.shape)