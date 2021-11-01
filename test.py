import torch
import torch.nn as nn

if __name__ == '__main__':
    x1 = torch.randn([3, 80])
    x2 = torch.randn([4, 3, 80])
    print(x2 + x1)