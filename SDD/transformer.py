import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import dataset, DataLoader
from torch.optim import Adam, SGD, RMSprop
from traj_dataset import TrajDataset

from einops import rearrange
from einops.layers.torch import Rearrange

from plotting import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is ", device)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_3d(patches, temperature=10000, dtype=torch.float32):
    _, f, h, w, dim, dtype = *patches.shape, patches.dtype

    x, y, z = torch.meshgrid(
        torch.arange(f, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            #nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim if out_dim else dim),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        print("after flatten: ", x.size())
        return self.net(x)


class TrajVit(nn.Module):

    def __init__(self, d_model, image_size, image_patch_size=(8, 8), frames=4, frame_patch_size=4, n_heads=8, n_layers=6,
                 mlp_dim=512, channels=1, batch_first=True):
        super(TrajVit, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.model_type = 'TrajVit'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # images and patches infos
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by the frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf)', p1=patch_height, p2=patch_width,
                      pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        ).to(self.device)

        # normalization
        self.normalisation = nn.Identity()

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=batch_first, device=device)
        self.encoder = TransformerEncoder(encoder_layer, n_layers)

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, batch_first=batch_first, device=device)
        self.decoder = TransformerDecoder(decoder_layer, n_layers)

        # inv normalization

        # MLP for regression
        input_mlp = num_patches * d_model
        print("input_mlp: ", input_mlp)
        self.mlp = FeedForward(input_mlp, mlp_dim, mlp_dim)
        return
    
    def forward(self, src, itm):
        *_, h, w, dtype = *src.shape, src.dtype

        src = src.to(self.device)
        itm = itm.to(self.device)

        print("src: ", src.size())
        print("itm: ", itm.size())

        src_e = self.to_patch_embedding(src).to(self.device)
        print("src after patch emb: ", src_e.size())
        src_pe = posemb_sincos_3d(src_e)
        src_e = rearrange(src_e, 'b ... d -> b (...) d') + src_pe

        itm_e = self.to_patch_embedding(itm).to(self.device)
        itm_pe = posemb_sincos_3d(itm_e)
        itm_e = rearrange(itm_e, 'b ... d -> b (...) d') + itm_pe

        print("src after emb: ", src_e.size())
        print("itm after emb: ", itm_e.size())

        x = self.encoder(src_e)
        print("after encoder: ", x.size())
        x = self.decoder(x, itm_e)
        print("before mlp: ", x.size())
        x = self.mlp(x)
        print("after mlp: ", x.size())
        return x


data_folder = "datasets/bookstore/video0/64_64_8"
dataset = TrajDataset(data_folder, n_prev=4, n_next=2, n_traj=1)
image_size = dataset.get_image_size()

model = TrajVit(512, image_size, frames=dataset.n_prev, frame_patch_size=2)

train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

optim = Adam(params=model.parameters(), lr=0.001)
criterion = nn.MSELoss()

loss_evol = []
n_epochs = 3
# Should add epochs
for epoch in range(n_epochs):

    epoch_loss = []
    for id_b, batch in enumerate(train_loader):
        src = batch['src']
        itm = batch['intermediate']
        tgt = batch['tgt']
        pred = model(src, itm)
        print("pred size: ", pred.size())
        print("tgt size: ", tgt.size())
        loss = criterion(pred, tgt)
        epoch_loss.append(loss.item())
        loss.backwards()
        optim.step()
        break
    loss_evol.append(epoch_loss)

plot_losses(loss_evol)


