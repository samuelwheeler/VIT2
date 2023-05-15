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
import random


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)


        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransposedAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 4, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(4, dim = -1)
        q, k, v, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q.transpose(-1,-2), k) * self.scale

        attn = self.attend(dots)
        
        v = v.transpose(-1,-2)

        v = torch.matmul(v, v2)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    
class QuadraticAttention(nn.Module):
    def __init__(self, dim, l,  heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        #inner_dim = dim_head *  heads
        #project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.middle_matrix = nn.Linear(dim, l, bias = False)

        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()

    def forward(self, x):
        out = torch.matmul(self.middle_matrix(x), x)
        return out

class AllRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        perm2 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        perm3 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        self.permq = torch.tensor(perm1, requires_grad = False)
        self.permk = torch.tensor(perm2, requires_grad = False)
        self.permv = torch.tensor(perm3, requires_grad = False)
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xq = flat_x[:, self.permq]
        flat_xk = flat_x[:, self.permk]
        flat_xv = flat_x[:, self.permv]


        shuffled_xq = flat_xq.view(x.size())
        shuffled_xk = flat_xk.view(x.size())
        shuffled_xv = flat_xv.view(x.size())


        q = self.to_q(shuffled_xq)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(shuffled_xk)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(shuffled_xv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class QKRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        perm2 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        self.permq = torch.tensor(perm1, requires_grad = False)
        self.permk = torch.tensor(perm2, requires_grad = False)
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xq = flat_x[:, self.permq]
        flat_xk = flat_x[:, self.permk]


        shuffled_xq = flat_xq.view(x.size())
        shuffled_xk = flat_xk.view(x.size())


        q = self.to_q(shuffled_xq)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(shuffled_xk)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class QRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        
        self.permq = torch.tensor(perm1, requires_grad = False)
        
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xq = flat_x[:, self.permq]


        shuffled_xq = flat_xq.view(x.size())
    


        q = self.to_q(shuffled_xq)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(x)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class KRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        
        self.permk = torch.tensor(perm1, requires_grad = False)
        
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xk = flat_x[:, self.permk]


        shuffled_xk = flat_xk.view(x.size())
    


        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(shuffled_xk)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class VRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        
        self.permv = torch.tensor(perm1, requires_grad = False)
        
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xv = flat_x[:, self.permv]


        shuffled_xv = flat_xv.view(x.size())
    


        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(x)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(shuffled_xv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class KVRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        
        self.permk = torch.tensor(perm1, requires_grad = False)
        self.permv = torch.tensor(perm1, requires_grad = False)
        
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xk = flat_x[:, self.permk]
        flat_xv = flat_x[:, self.permv]


        shuffled_xk = flat_xk.view(x.size())
        shuffled_xv = flat_xv.view(x.size())
    


        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(shuffled_xk)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(shuffled_xv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class QVRandomShuffleAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        perm1 = list(range(dim)) + random.sample(range(dim, l*dim), (l-1)*dim)
        
        self.permq = torch.tensor(perm1, requires_grad = False)
        self.permv = torch.tensor(perm1, requires_grad = False)
        
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        flat_x = x.flatten(start_dim=1)

        flat_xq = flat_x[:, self.permq]
        flat_xv = flat_x[:, self.permv]


        shuffled_xq = flat_xq.view(x.size())
        shuffled_xv = flat_xv.view(x.size())
    


        q = self.to_q(shuffled_xq)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        k = self.to_k(x)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)

        v = self.to_v(shuffled_xv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class ShuffleRowAttention(nn.Module):
    def __init__(self, dim, l, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        perm = [0] + random.sample(range(1, l), l-1)
        self.perm = torch.tensor(perm)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        q = q[:,:, self.perm]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class QuarticAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        #self.scale = dim_head ** -0.5

        #self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 5, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(5, dim = -1)
        q1, k1, q2, k2, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots1 = torch.matmul(q1, k1.transpose(-1, -2)) 
        dots2 = torch.matmul(q2, k2.transpose(-1, -2)) 
        dots = torch.matmul(dots1, dots2)
        #attn = self.attend(dots)

        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class NoSoftMaxAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        #self.scale = dim_head ** -0.5

        #self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) 

        #attn = self.attend(dots)

        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, attention_type, dim, depth, heads, dim_head, mlp_dim, num_patches, fixed_size, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        if attention_type == 'standard':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'quartic':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, QuarticAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'no_softmax':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, NoSoftMaxAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'row_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, ShuffleRowAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'allrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, AllRandomShuffleAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'qkrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, QKRandomShuffleAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'qrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, QRandomShuffleAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'quadratic':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, QuadraticAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'krandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, KRandomShuffleAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'vrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, VRandomShuffleAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'kvrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, KVRandomShuffleAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'qvrandom_shuffled':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, QVRandomShuffleAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, l = dim if fixed_size else num_patches + 1 )),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))
        elif attention_type == 'transposed':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ]))



    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, attention_type, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., fixed_size = False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.fixed_size = fixed_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        if fixed_size or attention_type == 'transposed':
            self.transformer = Transformer(attention_type = attention_type, dim = dim, depth = depth, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim,
                                        dropout = dropout, num_patches = num_patches, fixed_size = self.fixed_size)
        else:
            self.transformer = Transformer(attention_type = attention_type, dim = dim, depth = depth, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim,
                                        dropout = dropout, num_patches = num_patches, fixed_size = self.fixed_size)
        if attention_type == 'transposed':
            self.first = TransposedAttention(dim)
            self.first_transformer = Transformer(attention_type = 'standard', dim = dim, depth = 2, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim,
                                                 dropout = dropout, num_patches = num_patches, fixed_size = False)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        if self.fixed_size:
            self.first_transformer = Transformer(attention_type = 'standard', dim = dim, depth = 2, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim,
                                       dropout = dropout, num_patches = num_patches, fixed_size = False)
            #self.fix_length = nn.Linear(dim, 64, bias = False)
            self.fl_net = nn.Sequential(
                nn.Linear(dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 64),
                nn.Dropout(dropout)
            )
            self.fl_normer= nn.LayerNorm(dim)
            self.fl_next = nn.Sequential(
                nn.Linear(dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, dim),
                nn.Dropout(dropout)
            )
        
        self.atn_type = attention_type

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        
        
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(64 + 1)]
        x = self.dropout(x)
        if self.fixed_size:
            x = self.first_transformer(x)
            y = self.fl_net(x)
            x = torch.matmul(y.transpose(-1, -2),x)
            #x = self.fl_normer(x)
            #x = self.fl_next(x)
        if self.atn_type == 'transposed':
            x = self.first_transformer(x)
            x = self.first(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
