import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset, DataLoader
from torch.optim import Adam, SGD, RMSprop
from traj_dataset import TrajDataset


class TrajTransformer(nn.Module):
    
    def __init__(self, d_model, n_heads=8, n_layers=6):
        super(TrajTransformer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.model_type = 'Transformer'


        self.transformer = nn.Transformer()

        
        # layers
        self.tf = Transformer(d_model)
        return
    
    def forward(self, X):
        X = self.transformer()


if torch.cuda.is_available():
    device=torch.device("cuda")


dataset = TrajDataset("bookstore/video0")


model = TrajTransformer(512)

train_loader = DataLoader(dataset,batch_size=10,shuffle=True)


optim = Adam(lr=0.001)

# Should add epochs
for id_b, batch in enumerate(train_loader):

    optim.optimizer.zero_grad()

    print(batch)




