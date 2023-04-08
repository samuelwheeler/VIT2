import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., dim_head = 64):
        super().__init__()
        #dim_head = dim
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Sequential(
            #Rearrange('b c p -> b p c'),
            nn.Linear(dim, inner_dim * 3, bias = False)
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b p (h d) -> b h p d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h p d -> b p (h d)')
        out = self.to_out(out)
        return out
        #return rearrange(self.to_out(out), 'b p c -> b c p')

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            #Rearrange('b c p -> b p c'),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
            #Rearrange('b p c -> b c p')
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, hidden_dim, patch_size, heads = 8, dropout = 0. ):
        super().__init__()
        self.atn = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, hidden_dim, dropout)
        self.layer_norm = nn.LayerNorm([(32//pair(patch_size)[0] * 32//pair(patch_size)[1])+1, dim])
    def forward(self, x):
        x = self.layer_norm(self.atn(x) + x)
        x = self.layer_norm(self.ff(x) + x)

        return x




class Encodings(nn.Module):
    def __init__(self, *,  image_size, patch_size, dim, channels = 3, dim_head = 64, emb_dropout = 0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        patches_up, patches_across = (image_height // patch_height), (image_width // patch_width)

        patch_dim = channels * patch_height * patch_width

        assert dim >= patch_dim

        patch_embedding_layers = [Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)]

        if dim < patch_dim:
            patch_embedding_layers.append(nn.Linear(patch_dim, dim, bias = False))

        self.to_patch_embedding = nn.Sequential(*patch_embedding_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, patches_up * patches_across + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        
        self.dropout = nn.Dropout(emb_dropout)
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, p, c = x.shape
        cls_tokens = repeat(self.cls_token, '() p d -> b p d', b=b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(p + 1)]
        x = self.dropout(x)
        return x
        #return rearrange(x, 'b p c -> b c p')



class ViT_Model(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, hidden_dim,  numblocks, heads = 12, dim_head = 64, dropout = 0.0,channels = 3,  emb_dropout = 0., num_classes = 10):
        super().__init__()
        self.enc = Encodings(image_size = image_size, patch_size = patch_size, dim = dim, channels = channels)
        self.TB = nn.ModuleList([nn.Sequential(
            Transformer(dim = dim, hidden_dim = hidden_dim, patch_size=patch_size),
                nn.Dropout(p = 0.2)
            ) for i in range(numblocks)])
        self.final = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x):
        x = self.enc(x)
        for block in self.TB:
            x = block(x) + x
        x = self.final(x[:,0])
        return x
