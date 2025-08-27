# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.basic_modules.common import LayerNorm2d

class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att

class CrossAttentionYYX(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head

    def forward(self, x, cond_emb):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(cond_emb).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(cond_emb).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=256,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 diffusion_step=100,
                 timestep_type='adalayernorm'
                 ):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2   = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
        self.ln2_1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
        self.ln2_2 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
        self.ln3 = nn.LayerNorm(n_embd)

        # attention between image and pesudolabel.
        self.attn1 = CrossAttentionYYX(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        # self attention for input x.
        self.attn2 = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        # attention between pesudolabel and input x.
        self.attn3 = CrossAttentionYYX(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        # attention between image and input x.
        self.attn4 = CrossAttentionYYX(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x,  img_emb, plab_emb, timestep):

        # image refine pesudolabel.
        a, att = self.attn1(self.ln1(plab_emb), img_emb)
        plab_emb = plab_emb + a

        # self attention for x.
        a, att = self.attn2(self.ln2(x, timestep))
        x = x + a

        # pesudolabel guide the denoising process.
        a, att = self.attn3(self.ln2_1(x, timestep), plab_emb)
        x = x + a

        # image guide the denoizing process.
        a, att = self.attn4(self.ln2_2(x, timestep), img_emb)
        x = x + a

        x = x + self.mlp(self.ln3(x))

        return x, plab_emb


class Condition2LabelTransformer(nn.Module):
    def __init__(
            self,
            content_emb,
            plabel_emb,
            image_emb,
            init_conv,
            out_cls = 4,
            n_layer=14,
            n_embd=256,
            n_head=16,
            attn_pdrop=0,
            resid_pdrop=0,
            mlp_hidden_times=4,
            block_activate="GELU2",
            diffusion_step=100,
            timestep_type='adalayernorm',
            checkpoint=False,
    ):
        super().__init__()

        self.use_checkpoint = checkpoint
        self.out_cls = out_cls

        self.content_emb = content_emb
        self.plabel_emb = plabel_emb
        self.image_emb = image_emb
        self.init_conv = init_conv

        self.blocks = nn.Sequential(*[Block(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
            diffusion_step = diffusion_step,
            timestep_type = timestep_type,
        ) for n in range(n_layer)])

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(n_embd, n_embd // 4, kernel_size=2, stride=2),
            LayerNorm2d(n_embd // 4),  #  change for only channel 1
            nn.GELU(),
            nn.ConvTranspose2d(n_embd // 4, n_embd // 8, kernel_size=2, stride=2),
            LayerNorm2d(n_embd // 8),
            nn.GELU(),
            nn.ConvTranspose2d(n_embd // 8, n_embd // 16, kernel_size=2, stride=2),
            LayerNorm2d(n_embd // 16),
            nn.GELU(),
            nn.ConvTranspose2d(n_embd // 16, n_embd // 32, kernel_size=2, stride=2),
            nn.GELU(),
        )

        # final prediction head
        self.to_logits = nn.Sequential(
            LayerNorm2d(n_embd // 32),
            nn.Conv2d(n_embd // 32, out_cls, kernel_size=1, bias=False),
        )

        # ======================================================================================================
        self.upsample_pl = nn.Sequential(
            nn.ConvTranspose2d(n_embd, n_embd // 4, kernel_size=2, stride=2),
            LayerNorm2d(n_embd // 4),  # change for only channel 1
            nn.GELU(),
            nn.ConvTranspose2d(n_embd // 4, n_embd // 8, kernel_size=2, stride=2),
            LayerNorm2d(n_embd // 8),
            nn.GELU(),
            nn.ConvTranspose2d(n_embd // 8, n_embd // 16, kernel_size=2, stride=2),
            LayerNorm2d(n_embd // 16),
            nn.GELU(),
            nn.ConvTranspose2d(n_embd // 16, n_embd // 32, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.to_logits_pl = nn.Sequential(
            LayerNorm2d(n_embd // 32),
            nn.Conv2d(n_embd // 32, out_cls, kernel_size=1, bias=False),
        )
        # ======================================================================================================

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.005)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.005)
            if module.bias is not None:
                module.bias.data.zero_()

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        return super().parameters(recurse=True)

    def forward(
            self, 
            input,   # B x (192*192)
            img,     # B x (192*192)
            plabel,  # B x (192*192)
            t):

        ## =====================================================================================================
        # Stage 1: Extract patches from noisy_label x_t, condition_image y, pesudolabel x_p.
        cont_emb = self.content_emb(input)  # B x 256 x 12 x 12
        img_conv, pre_conv = self.init_conv(img)      # B x 64  x 192x 192
        img_emb = self.image_emb(img_conv)  # B x 256 x 12 x 12
        plab_emb = self.plabel_emb(plabel)  # B x 256 x 12 x 12

        # flatten and reshape
        b, c, h, w = cont_emb.shape

        emb = cont_emb.flatten(2).permute(0, 2, 1)       # B x 144 x 256
        img_emb = img_emb.flatten(2).permute(0, 2, 1)    # B x 144 x 256
        plab_emb = plab_emb.flatten(2).permute(0, 2, 1)  # B x 144 x 256
        ## =====================================================================================================

        ## =====================================================================================================
        # Stage 2: Interaction between cont_emb, img_emb, plab_emb, estimate denoized noisy_label embedding.
        for block_idx in range(len(self.blocks)):   
            if self.use_checkpoint == False:
                emb, plab_emb = self.blocks[block_idx](emb, img_emb, plab_emb, t.cuda()) # B x (Ld+Lt) x D, B x (Ld+Lt) x (Ld+Lt)
            else:
                emb, plab_emb = checkpoint(self.blocks[block_idx], emb, img_emb, plab_emb, t.cuda())
        ## =====================================================================================================

        ## =====================================================================================================
        # Stage 3: Upsample and conv to obtain the denoised noisy_label, x_t-1 or x_0.
        emb = emb.transpose(1, 2).view(b, c, h, w)  # B x 256 x 12 x 12

        emb = self.upsample(emb) # B x 8 x 192 x 192
        out = self.to_logits(emb) # B x out_cls x 192 x 192
        ## =====================================================================================================

        # ======================================================================================================
        plab_emb = plab_emb.transpose(1, 2).view(b, c, h, w)  # B x 256 x 12 x 12

        plab_emb = self.upsample_pl(plab_emb)  # B x 8 x 192 x 192
        out_pl = self.to_logits_pl(plab_emb)  # B x out_cls x 192 x 192
        # ======================================================================================================

        # output: B x out_cls x (192*192)
        return out.view(b, self.out_cls, -1), out_pl.view(b, self.out_cls, -1), img_conv, pre_conv