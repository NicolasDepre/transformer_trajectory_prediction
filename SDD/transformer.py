import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class TrajTransformer(nn.Module):
    
    def __init__(self, d_model, n_heads=8, n_layers=6):
        super(TrajTransformer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.model_type = 'Transformer'
        
        # layers
        self.tf = Transformer(d_model)
        return
    
    def forward(self, X):
        return self.tf(X)
    
    def train(self, X, Y):
        return
