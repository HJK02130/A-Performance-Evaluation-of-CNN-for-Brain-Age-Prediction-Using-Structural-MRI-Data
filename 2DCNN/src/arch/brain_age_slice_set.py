"""code for attention models"""

import math

import torch
from box import Box
from torch import nn


class MeanPool(nn.Module):
    def forward(self, X):
        return X.mean(dim=1, keepdim=True), None


class MaxPool(nn.Module):
    def forward(self, X):
        return X.max(dim=1, keepdim=True)[0], None


class PooledAttention(nn.Module):
    def __init__(self, input_dim, dim_v, dim_k, num_heads, ln=False):
        super(PooledAttention, self).__init__()
        self.S = nn.Parameter(torch.zeros(1, dim_k))
        nn.init.xavier_uniform_(self.S)

        # transform to get key and value vector
        self.fc_k = nn.Linear(input_dim, dim_k)
        self.fc_v = nn.Linear(input_dim, dim_v)

        self.dim_v = dim_v
        self.dim_k = dim_k
        self.num_heads = num_heads
        
        if ln:
            self.ln0 = nn.LayerNorm(dim_v)

    def forward(self, X):
        B, C, H = X.shape

        Q = self.S.repeat(X.size(0), 1, 1)

        K = self.fc_k(X.reshape(-1, H)).reshape(B, C, self.dim_k)
        V = self.fc_v(X.reshape(-1, H)).reshape(B, C, self.dim_v)
        dim_split = self.dim_v // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), 2)
        O = torch.cat(A.bmm(V_).split(B, 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        return O, A

    def get_attention(self, X):
        B, C, H = X.shape

        Q = self.S.repeat(X.size(0), 1, 1)

        K = self.fc_k(X.reshape(-1, H)).reshape(B, C, self.dim_k)
        V = self.fc_v(X.reshape(-1, H)).reshape(B, C, self.dim_v)
        dim_split = self.dim_v // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), 2)
        return A


def encoder_blk(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
        nn.InstanceNorm2d(out_channels),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU()
    )


class MRI_ATTN(nn.Module):

    def __init__(self, attn_num_heads, attn_dim, attn_drop=False, agg_fn="attention", slice_dim=1,
                 *args, **kwargs):
        super(MRI_ATTN, self).__init__()

        self.input_dim = [(1, 109, 91), (91, 1, 91), (91, 109, 1)][slice_dim - 1]

        self.num_heads = attn_num_heads
        self.attn_dim = attn_dim

        # Build Encoder
        encoder_blocks = [
                encoder_blk(1, 32),
                encoder_blk(32, 64),
                encoder_blk(64, 128),
                encoder_blk(128, 256),
                encoder_blk(256, 256)
        ]
        self.encoder = nn.Sequential(*encoder_blocks)

        if slice_dim == 1:
            avg = nn.AvgPool2d([3, 2])
        elif slice_dim == 2:
            avg = nn.AvgPool2d([2, 2])
        elif slice_dim == 3:
            avg = nn.AvgPool2d([2, 3])
        else:
            raise Exception("Invalid slice dim")
        self.slice_dim = slice_dim

        # Post processing
        self.post_proc = nn.Sequential(
            nn.Conv2d(256, 64, 1, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            avg,
            nn.Dropout(p=0.5) if attn_drop else nn.Identity(),
            nn.Conv2d(64, self.num_heads * self.attn_dim, 1)
        )

        if agg_fn == "attention":
            self.pooled_attention = PooledAttention(input_dim=self.num_heads * self.attn_dim,
                                                    dim_v=self.num_heads * self.attn_dim,
                                                    dim_k=self.num_heads * self.attn_dim,
                                                    num_heads=self.num_heads)
        elif agg_fn == "mean":
            self.pooled_attention = MeanPool()
        elif agg_fn == "max":
            self.pooled_attention = MaxPool()
        else:
            raise Exception("Invalid attention function")

        # Build regressor
        self.attn_post = nn.Linear(self.num_heads * self.attn_dim, 64)
        self.regressor = nn.Sequential(nn.ReLU(), nn.Linear(64, 1))
        self.init_weights()

    def init_weights(self):
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and "regressor" in k:
                m.bias.data.fill_(62.68)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):

        B, C, H, W, D = x.size()
        if self.slice_dim == 1:
            new_input = torch.cat([x[:, :, i, :, :] for i in range(H)], dim=0)
            encoding = self.encoder(new_input)
            encoding = self.post_proc(encoding)
            encoding = torch.cat([i.unsqueeze(2) for i in torch.split(encoding, B, dim=0)], dim=2)
            # note: squeezing is bad because batch dim can be dropped
            encoding = encoding.squeeze(4).squeeze(3)
        elif self.slice_dim == 2:
            new_input = torch.cat([x[:, :, :, i, :] for i in range(W)], dim=0)
            encoding = self.encoder(new_input)
            encoding = self.post_proc(encoding)
            encoding = torch.cat([i.unsqueeze(3) for i in torch.split(encoding, B, dim=0)], dim=3)
            # note: squeezing is bad because batch dim can be dropped
            encoding = encoding.squeeze(4).squeeze(2)
        elif self.slice_dim == 3:
            new_input = torch.cat([x[:, :, :, :, i] for i in range(D)], dim=0)
            encoding = self.encoder(new_input)
            encoding = self.post_proc(encoding)
            encoding = torch.cat([i.unsqueeze(4) for i in torch.split(encoding, B, dim=0)], dim=4)
            # note: squeezing is bad because batch dim can be dropped
            encoding = encoding.squeeze(3).squeeze(2)
        else:
            raise Exception("Invalid slice dim")

        # swap dims for input to attention
        encoding = encoding.permute((0, 2, 1))
        encoding, attention = self.pooled_attention(encoding)
        return encoding.squeeze(1), attention

    def forward(self, x):
        embedding, attention = self.encode(x)
        post = self.attn_post(embedding)
        y_pred = self.regressor(post)
        return Box({"y_pred": y_pred, "attention": attention})

    def get_attention(self, x):
        _, attention = self.encode(x)
        return attention


def get_arch(*args, **kwargs):
    return {"net": MRI_ATTN(*args, **kwargs)}
