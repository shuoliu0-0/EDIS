import torch
import torch.nn as nn
from einops import reduce
class Intention(nn.Module):

    def __init__(self, dim, num_heads, kqv_bias=False, device='cuda'):
        super(Intention, self).__init__()
        self.dim = dim
        self.head = num_heads
        self.head_dim = dim // num_heads
        self.device = device
        self.alpha = nn.Parameter(torch.rand(1))
        assert dim % num_heads == 0, 'dim must be divisible by num_heads!'

        self.wq = nn.Linear(dim, dim, bias=kqv_bias)
        self.wk = nn.Linear(dim, dim, bias=kqv_bias)
        self.wv = nn.Linear(dim, dim, bias=kqv_bias)

        self.softmax = nn.Softmax(dim=-2)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, query=None):
        if query is None:
            query = x

        query = self.wq(query)
        key = self.wk(x)
        value = self.wv(x)

        b, n, c = x.shape
        key = key.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)
        key_t = key.clone().permute(0, 1, 3, 2)
        value = value.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)

        b, n, c = query.shape
        query = query.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)

        kk = key_t @ key
        kk = self.alpha * torch.eye(kk.shape[-1], device=self.device) + kk
        kk_inv = torch.inverse(kk)
        attn_map = (kk_inv @ key_t) @ value

        attn_map = self.softmax(attn_map)

        out = (query @ attn_map)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        out = self.out(out)

        return out

class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.):
        super(SelfAttention, self).__init__()
        self.wq = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wk = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wv = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, x):
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        att, _ = self.attn(query, key, value)
        out = att + x
        return out

class IntentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, kqv_bias=True, device='cuda'):
        super(IntentionBlock, self).__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.attn = Intention(dim=dim, num_heads=num_heads, kqv_bias=kqv_bias, device=device)
        self.softmax = nn.Softmax(dim=-2)
        self.beta = nn.Parameter(torch.rand(1))

    def forward(self, x, q):
        x = self.norm_layer(x)
        q_t = q.permute(0, 2, 1)
        att = self.attn(x, q)
        att_map = self.softmax(att)
        out = self.beta * q_t @ att_map
        return out

class BiIntention(nn.Module):
    def __init__(self, embed_dim, layer=1, num_head=8, device='cuda'):
        super(BiIntention, self).__init__()

        self.layer = layer
        self.drug_intention = nn.ModuleList([
            IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        self.protein_intention = nn.ModuleList([
            IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        #self attention
        self.attn_drug = SelfAttention(dim=embed_dim, num_heads=num_head)
        self.attn_protein = SelfAttention(dim=embed_dim, num_heads=num_head)

    def forward(self, drug, protein):
        drug = self.attn_drug(drug)
        protein = self.attn_protein(protein)

        for i in range(self.layer):
            v_p = self.drug_intention[i](drug, protein)
            v_d = self.protein_intention[i](protein, drug)
            drug, protein = v_d, v_p

        v_d = reduce(drug, 'B H W -> B H', 'max')
        v_p = reduce(protein, 'B H W -> B H', 'max')

        f = torch.cat((v_d, v_p), dim=1)

        return f, v_d, v_p, None
