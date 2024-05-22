import torch
import torch.nn as nn
import torch.nn.functional as F

class CRReLU(nn.Module):
    def __init__(self,lr=0.01):
        super(CRReLU, self).__init__()
        self.lr=nn.Parameter(torch.tensor(lr))
    def forward(self,x):
        return F.relu(x)+self.lr*x*torch.exp(-x**2/2)
