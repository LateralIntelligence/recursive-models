'''
TRM (Tiny Recursive Reasoning Model) backbone for flow-matching LM.

Ports the inner recursive-reasoning core of TRM
(https://github.com/SamsungSAILMontreal/TinyRecursiveModels) and wraps it to
match the denoiser/backbone contract used by the DiT backbone:

    forward(xt, sigma, sigma_prime=None, use_jvp_attn=False) -> logits (B, L, vocab)

The ACT wrapper, cross-call carry, halting Q-head, and puzzle/sparse embeddings
of the original are intentionally dropped -- a flow-matching denoiser is
stateless per call. Time conditioning (absent in TRM) is injected additively:
a TimestepEmbedder maps sigma into hidden-size space and is added to the token
embeddings, so it enters every L-cycle through ``z_H + input_embeddings``.
'''
import math

import einops
import flash_attn
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit import EmbeddingLayer, TimestepEmbedder, Rotary, apply_rotary_pos_emb


def rms_norm(x, variance_epsilon=1e-5):
    """RMSNorm computed in float32 then cast back (no learned weight, as in TRM)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    return x.to(input_dtype)


def _find_multiple(a, b):
    return (-(a // -b)) * b


class SwiGLU(nn.Module):
    def __init__(self, dim, expansion):
        super().__init__()
        inter = _find_multiple(round(expansion * dim * 2 / 3), 256)
        self.gate_up_proj = nn.Linear(dim, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TRMBlock(nn.Module):
    """Post-norm TRM block: non-causal rotary attention + SwiGLU MLP."""

    def __init__(self, dim, num_heads, expansion, rms_norm_eps):
        super().__init__()
        assert dim % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.norm_eps = rms_norm_eps
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.mlp = SwiGLU(dim, expansion)

    def _attn(self, x, rotary_cos_sin): #TODO: double check this
        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        # Mirror DDiTBlock: apply rotary in fp32, then flash qkv-packed attention.
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype), use_flash=True)
        x = flash_attn.flash_attn_qkvpacked_func(qkv, 0.0, causal=False)
        x = einops.rearrange(x, 'b s h d -> b s (h d)')
        return self.attn_out(x)

    def forward(self, hidden_states, rotary_cos_sin):
        hidden_states = rms_norm(
            hidden_states + self._attn(hidden_states, rotary_cos_sin),
            variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps)
        return hidden_states


class ReasoningModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states, input_injection, rotary_cos_sin):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_cos_sin)
        return hidden_states


class TRM(nn.Module):
    """TRM recursive-reasoning core as a flow-matching denoiser backbone."""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        self.vocab_size = vocab_size

        dim = config.model.hidden_size
        num_heads = getattr(config.model, 'trm_num_heads', config.model.n_heads)
        self.num_heads = num_heads
        self.H_cycles = config.model.H_cycles
        self.L_cycles = config.model.L_cycles
        L_layers = config.model.L_layers
        expansion = config.model.expansion
        rms_norm_eps = getattr(config.model, 'rms_norm_eps', 1e-5)
        rope_theta = getattr(config.model, 'rope_theta', 10000.0)

        self.embed_scale = math.sqrt(dim)
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)

        # Time (noise level) conditioning -- additive into hidden-size space.
        self.sigma_map = TimestepEmbedder(dim)
        if getattr(config.algo, 'double_temb', False):
            self.sigma_map_prime = TimestepEmbedder(dim)
        else:
            self.sigma_map_prime = None

        self.rotary_emb = Rotary(dim // num_heads, base=rope_theta)

        self.L_level = ReasoningModule(
            [TRMBlock(dim, num_heads, expansion, rms_norm_eps)
             for _ in range(L_layers)])

        # Fixed initial reasoning states (buffers, as in TRM -- not trained).
        self.register_buffer(
            'H_init', nn.init.trunc_normal_(torch.empty(dim), std=1.0), persistent=True)
        self.register_buffer(
            'L_init', nn.init.trunc_normal_(torch.empty(dim), std=1.0), persistent=True)

        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight.data.zero_()  # zero-init, matching DDiTFinalLayer

    def forward(self, x, sigma, sigma_prime=None, use_jvp_attn=False):
        if use_jvp_attn:
            raise NotImplementedError(
                "TRM backbone does not support JVP attention (use_jvp_attn=True). "
                "Use backbone=dit for JVP-based distillation methods.")

        input_embeddings = self.vocab_embed(x) * self.embed_scale  # (B, L, D)

        t_emb = self.sigma_map(sigma)
        if sigma_prime is not None:
            t_prime_emb = (self.sigma_map_prime or self.sigma_map)(sigma_prime)
            t_emb = t_emb + t_prime_emb
        input_embeddings = input_embeddings + F.silu(t_emb)[:, None, :]

        B, L, D = input_embeddings.shape
        z_H = self.H_init.view(1, 1, D).expand(B, L, D)
        z_L = self.L_init.view(1, 1, D).expand(B, L, D)

        rotary_cos_sin = self.rotary_emb(input_embeddings)

        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            # TRM deep supervision: all but the final cycle run without gradient.
            with torch.no_grad():
                for _H_step in range(self.H_cycles - 1):
                    for _L_step in range(self.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_embeddings, rotary_cos_sin)
                    z_H = self.L_level(z_H, z_L, rotary_cos_sin)
            # Final cycle carries the 1-step gradient.
            for _L_step in range(self.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, rotary_cos_sin)
            z_H = self.L_level(z_H, z_L, rotary_cos_sin)

            logits = self.lm_head(z_H)  # (B, L, vocab_size)

        return logits
