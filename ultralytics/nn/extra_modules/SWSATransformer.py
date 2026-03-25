import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

__all__ = ['TransformerEncoderLayer_SWSA']


class LayerNorm(nn.Module):
    """LayerNorm that supports channels_first data format."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PatchSA(nn.Module):
    """Patch Self-Attention module."""
    def __init__(self, dim, heads=8, patch_size=4, stride=4):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.patch_size = patch_size
        self.stride = stride

        self.to_qkv = nn.Conv2d(dim * 3, dim * 3, 1, groups = dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

        self.pos_encode = nn.Parameter(torch.zeros((2 * patch_size - 1) ** 2, heads))
        trunc_normal_(self.pos_encode, std=0.02)
        coord = torch.arange(patch_size)
        coords = torch.stack(torch.meshgrid([coord, coord], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += patch_size - 1
        relative_coords[:, :, 1] += patch_size - 1
        relative_coords[:, :, 0] *= 2 * patch_size - 1
        pos_index = relative_coords.sum(-1)
        self.register_buffer('pos_index', pos_index)

    def _forward1(self, x):
        B, C, H, W = x.shape
        assert H == W
        res = torch.empty_like(x)
        pad_num = self.patch_size - self.stride
        expan_x = F.pad(x, (0,pad_num,0,pad_num), mode='replicate')
        repeat_x = [expan_x] * 3
        expan_x = torch.cat(repeat_x, dim=1)
        qkv = self.to_qkv(expan_x)

        for i in range(0, H, self.stride):
            for j in range(0, W, self.stride):
                patch = qkv[:, :, i: i + self.patch_size, j: j + self.patch_size]
                patch = patch.reshape(B, 3, self.heads, -1, self.patch_size ** 2).permute(1, 0, 2, 4, 3)
                q, k, v = patch[0], patch[1], patch[2]
                q = q * self.scale
                attn = (q @ k.transpose(-2, -1))

                pos_encode = self.pos_encode[self.pos_index.view(-1)].view(
                    self.patch_size ** 2, self.patch_size ** 2, -1)
                pos_encode = pos_encode.permute(2, 0, 1).contiguous()
                attn = attn + pos_encode.unsqueeze(0)

                attn = self.softmax(attn)
                _res = (attn @ v)
                _res = _res.transpose(-2, -1).reshape(B, -1, self.patch_size, self.patch_size)

                res[:, :, i: i + self.stride, j: j + self.stride] = _res[:, :, :self.stride, :self.stride]

        return self.to_out(res)

    def _forward2(self, x):
        B, C, H, W = x.shape
        assert H == W
        pad_num = self.patch_size - self.stride
        patch_num = ((H + pad_num - self.patch_size) // self.stride + 1) ** 2
        expan_x = F.pad(x, (0,pad_num,0,pad_num), mode='replicate')
        repeat_x = [expan_x] * 3
        expan_x = torch.cat(repeat_x, dim=1)
        qkv = self.to_qkv(expan_x)

        qkv_patches = F.unfold(qkv, kernel_size=self.patch_size, stride=self.stride)
        qkv_patches = qkv_patches.view(B, 3, self.heads, -1, self.patch_size**2, patch_num).permute(1, 0, 2, 5, 4, 3)
        q, k, v = qkv_patches[0], qkv_patches[1], qkv_patches[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        pos_encode = self.pos_encode[self.pos_index.view(-1)].view(self.patch_size ** 2, self.patch_size ** 2, -1)
        pos_encode = pos_encode.permute(2, 0, 1).contiguous().unsqueeze(1).repeat(1,patch_num,1,1)
        attn = attn + pos_encode.unsqueeze(0)

        attn = self.softmax(attn)
        _res = (attn @ v)

        _res = _res.view(B, self.heads, patch_num, self.patch_size, self.patch_size, -1)[:, :, :, :self.stride, :self.stride]
        _res = _res.transpose(2,5).contiguous().view(B,-1,patch_num)
        res = F.fold(_res, output_size=(H, W), kernel_size=self.stride, stride=self.stride)
        return self.to_out(res)

    def forward(self, x):
        # return self._forward1(x) # this version is easier to understand while slower
        return self._forward2(x) # this version is faster


class TransformerEncoderLayer_SWSA(nn.Module):
    """Defines a single layer of the transformer encoder with SWSA."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.SWSA = PatchSA(c1, heads=num_heads)
        # Feedforward model
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)

        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        src2 = self.SWSA(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
