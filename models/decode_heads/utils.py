from torch import nn
from torch import Tensor
from typing import Optional
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from timm.models.layers import trunc_normal_


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv, return_memory=False):
        B, Nq, C = xq.size()  # 1, 21, 512
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                               C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                               C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                               C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_memory:
            return x, attn_save.sum(dim=1) / self.num_heads, k.permute(
                0, 2, 1, 3).reshape(B, Nk, C), v.permute(0, 2, 1,
                                                         3).reshape(B, Nv, C)
        return x, attn_save.sum(dim=1) / self.num_heads


class TPN_Decoder(TransformerDecoder):

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                return_memory=False):
        if return_memory:
            output = tgt
            attns = []
            outputs = []
            ks = []
            vs = []
            for mod in self.layers:
                output, attn, k, v = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_memory=True)
                attns.append(attn)
                outputs.append(output)
                ks.append(k)
                vs.append(v)

            if self.norm is not None:  # not do
                output = self.norm(output)

            return outputs, attns, ks, vs
        else:
            output = tgt
            attns = []
            outputs = []
            for mod in self.layers:
                output, attn = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask)
                attns.append(attn)
                outputs.append(output)
            if self.norm is not None:  # not do
                output = self.norm(output)

            return outputs, attns


class TPN_DecoderLayer(TransformerDecoderLayer):

    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(kwargs['d_model'],
                                        num_heads=kwargs['nhead'],
                                        qkv_bias=True,
                                        attn_drop=0.1)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                return_memory=False) -> Tensor:
        if return_memory:
            tgt0, attn, k, v = self.multihead_attn(tgt,
                                                   memory,
                                                   memory,
                                                   return_memory=True)
            tgt = tgt + self.dropout2(tgt0)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(
                self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt, attn, k, v
        else:
            tgt0, attn = self.multihead_attn(tgt, memory, memory)
            tgt = tgt + self.dropout2(tgt0)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(
                self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt


class RecoveryDecoder(nn.Module):

    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.decouple_q = nn.ModuleList()
        self.decouple_v = nn.ModuleList()
        for i in range(num_layers):
            self.decouple_q.append(
                TPN_DecoderLayer(d_model=dim,
                                 nhead=nhead,
                                 dim_feedforward=dim * 4))
            self.decouple_v.append(
                TPN_DecoderLayer(d_model=dim,
                                 nhead=nhead,
                                 dim_feedforward=dim * 4))
        self.linear_q_in = nn.Linear(dim, dim)
        self.linear_k_in = nn.Linear(dim, dim)
        self.linear_q_out = nn.Linear(dim, dim * 2)
        self.linear_k_out = nn.Linear(dim, dim)
        self.init_weights()

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, q, lateral):
        q = self.linear_q_in(q)
        lateral = self.linear_k_in(lateral)
        for i in range(self.num_layers):
            q, lateral = self.decouple_q[i](q, lateral), self.decouple_v[i](
                lateral, q)
        q = self.linear_q_out(q)
        lateral = self.linear_k_out(lateral)
        return q, lateral
