import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from traj_dataset import TrajDataset
from torch.utils.data import DataLoader
from torch.optim import Adam

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MyTransformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dim_feedforward, batch_first=True, device=None):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, batch_first=batch_first, device=device)
        self.encoder = nn.TransformerEncoder(enc_layer, n_heads)

    def forward(self, x, y):
        print("input shape : ", x.shape)
        x = self.encoder(x)
        print("after encoder : ", x.shape)
        return x


class SimpleViT(nn.Module):
    def __init__(self, *, image_size=(136, 178), patch_size=(68, 89), dim=512, depth=6, heads=8, mlp_dim=32, output_dim=4, channels=1, dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        print(patch_height, patch_width)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        print(num_patches)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b p (h p1) (w p2) -> b p h w (p1 p2)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        #self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        #self.transformer = nn.Transformer(d_model=dim, batch_first=True)
        self.transformer = MyTransformer(d_model=dim, n_layers=depth, n_heads=heads, dim_feedforward=2048, batch_first=True)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim)
        )

    def forward(self, img, tgt):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        print("after embedding : ", x.shape)
        x = rearrange(x, 'b p ... d -> b p (...) d')
        print("after rearrange : ", x.shape)
        x = x + pe

        x = self.transformer(x, tgt)
        x = self.to_latent(x)  # doesn't do anything as
        return self.linear_head(x)


dataset = TrajDataset("datasets/bookstore/video0")


model = SimpleViT(dim=512)
optim = Adam(params=model.parameters(), lr=0.001)


train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Should add epochs
for id_b, batch in enumerate(train_loader):
    X = batch["X"]
    Y = batch["Y"]
    print(X.shape)
    print(Y.shape)
    pred = model(X, Y)
    print(pred.shape)
    print(pred[0])
    print("Si ca passe je suis un PDDDD (tour montparnasse) ")
    #optim.zero_grad()
    break
