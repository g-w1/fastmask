# %%
import copy
import inspect
import math
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional
from param_level_masking import RoutedLinear as RoutedLinearFn

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from jaxtyping import Float, Int


@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 8
    n_head: int = 12
    n_key_value_head: Optional[int] = 2
    n_embd: int = 768
    mlp_dims: int = 768 * 4
    expand_dims: int = 64
    attn_bias: bool = True
    tie_weights: bool = False
    init_weights: bool = True


def contract_module(module: nn.Module) -> nn.Module:
    """
    If the module has a contract method, call it and return the result.
    Otherwise, create a new module with the same parameters and return it.
    """
    if hasattr(module, "contract"):
        return module.contract()

    new_module = copy.deepcopy(module)
    with torch.no_grad():
        for new_p, old_p in zip(new_module.parameters(), module.parameters()):
            new_p[:] = old_p[:]
    return new_module


# from https://github.com/KellerJordan/modded-nanogpt/blob/e48427208f14a0a27ef0fcf2d369a0f4299342d7/train_gpt.py#L144C1-L161C49
class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=dim // 4, dtype=torch.float32
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x):
        cos, sin = (
            self.cos[None, : x.size(-3), None, :],
            self.sin[None, : x.size(-3), None, :],
        )
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class CausalGroupedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_key_value_head = config.n_key_value_head
        self.d_head = config.n_embd // config.n_head

        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_key_value_head == 0

        self.c_attn_q = nn.Linear(
            config.n_embd, 1 * config.n_embd, bias=config.attn_bias
        )
        self.c_attn_kv = nn.Linear(
            config.n_embd,
            2 * config.n_key_value_head * self.d_head,
            bias=config.attn_bias,
        )  # * 2 b/c we want key and value projs
        self.rotary = Rotary(
            config.n_embd // config.n_head
        )  # dim // num_heads = head_dim
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)

    def doubled_retain_params(self) -> list:
        return []

    def doubled_forget_params(self) -> list:
        return []

    def forward(self, x, tok_masks):
        B, T, C = x.size()

        q = self.c_attn_q(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k, v = self.c_attn_kv(x).split(self.n_key_value_head * self.d_head, dim=2)
        k = k.view(B, T, self.n_key_value_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_key_value_head, self.d_head).transpose(1, 2)
        # https://github.com/KellerJordan/modded-nanogpt/blob/e48427208f14a0a27ef0fcf2d369a0f4299342d7/train_gpt.py#L187
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, torch.tensor(0.0, device=x.device)

    def forward_ablated(self, x):
        return self.forward(x, None)[0]

    def magnitude_of_proj_up_matrix(self):
        return self.c_fc.weight.abs().sum() + self.c_fc.bias.abs().sum()


# https://github.com/KellerJordan/modded-nanogpt/blob/e48427208f14a0a27ef0fcf2d369a0f4299342d7/train_gpt.py#L133-L134
def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.normalized_shape,), self.weight, self.eps)


class RoutedTransformer(nn.Module):
    def __init__(
        self,
        config,
        embedding: nn.Module,
        unembedding: nn.Module,  # to tie the weights, just tie before passing it in
        blocks: list,
    ):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "embd": embedding,
                "blocks": nn.ModuleList(blocks),
                "ln_f": RMSNorm(config.n_embd),
                "unembd": unembedding,
            }
        )

        if config.init_weights:
            # init all weights
            self.apply(self._init_weights)
            # apply special scaled init to the residual projections, per GPT-2 paper
            for pn, p in self.named_parameters():
                if pn.endswith("c_proj.weight"):
                    # https://github.com/KellerJordan/modded-nanogpt/blob/e48427208f14a0a27ef0fcf2d369a0f4299342d7/train_gpt.py#L249
                    p.data.zero_()

    def doubled_retain_params(self) -> list:
        retain_params = []
        for block in self.transformer.blocks:
            retain_params.extend(block.doubled_retain_params())
        return retain_params

    def doubled_forget_params(self) -> list:
        forget_params = []
        for block in self.transformer.blocks:
            forget_params.extend(block.doubled_forget_params())
        return forget_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, toks, targets=None, mask_ids=None, reduce_loss=True, stop_at_layer=None
    ):
        device = toks.device
        b, t = toks.size()

        x, aux_loss = self.transformer.embd(toks, mask_ids)
        for i, block in enumerate(self.transformer.blocks):
            x, aux_loss_block = block(x, mask_ids)
            aux_loss = aux_loss + aux_loss_block
            if i == stop_at_layer:
                return x, aux_loss
        x = self.transformer.ln_f(x)
        logits, aux_loss_unemb = self.transformer.unembd(x, mask_ids)
        aux_loss = aux_loss + aux_loss_unemb
        lm_loss = torch.tensor(0.0, device=device)
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction="mean" if reduce_loss else "none",
            )
        return logits, aux_loss, lm_loss

    def forward_ablated(self, toks, targets=None, reduce_loss=True, stop_at_layer=None):
        x = self.transformer.embd.forward_ablated(toks)
        for i, block in enumerate(self.transformer.blocks):
            x = block.forward_ablated(x)
            if i == stop_at_layer:
                return x, torch.tensor(0.0, device=x.device)
        x = self.transformer.ln_f(x)
        logits = self.transformer.unembd.forward_ablated(x)
        lm_loss = torch.tensor(0.0, device=x.device)
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction="mean" if reduce_loss else "none",
            )
        return logits, lm_loss

    # TODO deal with the fact that we want to decay based on how much it was used / it's gradients
    # Look into how MOEs do this
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        total_num_in_millions = (num_decay_params + num_nodecay_params) / 1_000_000
        print(f"total number of parameters: {total_num_in_millions:.2f}M")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

        return optimizer

    def contract(self) -> "RoutedTransformer":
        contracted_blocks = [
            contract_module(block) for block in self.transformer.blocks
        ]
        contracted_embd = contract_module(self.transformer.embd)
        contracted_unembd = contract_module(self.transformer.unembd)
        new_config = copy.deepcopy(self.config)
        new_config.init_weights = False
        new_routed_transformer = RoutedTransformer(
            new_config,
            contracted_embd,
            contracted_unembd,
            contracted_blocks,
        )
        new_routed_transformer.transformer.ln_f = contract_module(self.transformer.ln_f)
        return new_routed_transformer

    @torch.no_grad()
    def generate(self, prompt_tokenized, max_new_toks, temperature):
        device = next(iter(self.parameters())).device
        prompt_tokenized = (
            torch.tensor(prompt_tokenized).unsqueeze(0)
            if isinstance(prompt_tokenized, list)
            else prompt_tokenized
        ).to(device)
        for _ in range(max_new_toks):
            logits = self.forward(prompt_tokenized)[0]
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            prompt_tokenized = torch.cat([prompt_tokenized, next_token], dim=1)
        return prompt_tokenized


class Block(nn.Module):
    def __init__(
        self,
        config,
        attention: nn.Module,
        mlp: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.attn = attention
        self.mlp = mlp
        self.rmsnorm_1 = RMSNorm(config.n_embd)
        self.rmsnorm_2 = RMSNorm(config.n_embd)

    def forward(self, x, mask_ids):
        attn_out, aux_loss_attn = self.attn(self.rmsnorm_1(x), mask_ids)
        x = x + attn_out
        mlp_out, aux_loss_mlp = self.mlp(self.rmsnorm_2(x), mask_ids)
        x = x + mlp_out
        return x, aux_loss_attn + aux_loss_mlp

    def doubled_retain_params(self) -> list:
        return self.attn.doubled_retain_params() + self.mlp.doubled_retain_params()

    def doubled_forget_params(self) -> list:
        return self.attn.doubled_forget_params() + self.mlp.doubled_forget_params()

    def forward_ablated(self, x):
        attn_out = self.attn.forward_ablated(self.rmsnorm_1(x))
        x = x + attn_out
        mlp_out = self.mlp.forward_ablated(self.rmsnorm_2(x))
        x = x + mlp_out
        return x

    def contract(self) -> "Block":
        contracted_attn = contract_module(self.attn)
        contracted_mlp = contract_module(self.mlp)
        new_block = Block(self.config, contracted_attn, contracted_mlp)
        new_block.rmsnorm_1 = contract_module(self.rmsnorm_1)
        new_block.rmsnorm_2 = contract_module(self.rmsnorm_2)
        return new_block


class Embd(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)

    def forward(self, x, mask_ids):
        return self.embedding(x), torch.tensor(0.0, device=x.device)

    def forward_ablated(self, x):
        return self.embedding(x)


class UnEmbd(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.unembedding = nn.Linear(config.n_embd, config.vocab_size, bias=True)

    def forward(self, x, mask_ids):
        return self.unembedding(x), torch.tensor(0.0, device=x.device)

    def forward_ablated(self, x):
        return self.unembedding(x)


routed_linear = RoutedLinearFn.apply


class RoutedLinear(nn.Module):
    def __init__(self, in_features, out_features, masking_is_param_level, bias=True):
        super().__init__()
        self.masking_is_param_level = masking_is_param_level
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(self, x, lr_mults):
        return routed_linear(
            x,
            self.weight,
            self.bias,
            lr_mults,
            self.masking_is_param_level,
        )


class RegularMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        expanded_size = config.mlp_dims
        self.c_fc = nn.Linear(config.n_embd, expanded_size, bias=True)
        self.c_proj = nn.Linear(expanded_size, config.n_embd, bias=True)

    def forward(self, x, tok_masks):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x, torch.tensor(0.0, device=x.device)

    def forward_ablated(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

    def contract(self) -> "RegularMLP":
        contracted = RegularMLP(self.config)
        contracted.c_fc.weight[:] = self.c_fc.weight[:]
        contracted.c_fc.bias[:] = self.c_fc.bias[:]
        contracted.c_proj.weight[:] = self.c_proj.weight[:]
        contracted.c_proj.bias[:] = self.c_proj.bias[:]
        return contracted

    def magnitude_of_proj_up_matrix(self):
        return self.c_fc.weight.abs().sum() + self.c_fc.bias.abs().sum()


@dataclass
class MLPOrResidualMaskConfig:
    dimensions: list[int]
    pos_lr: float
    neg_lr: float


class DiscreteMaskingMLP(nn.Module):
    def __init__(self, config, mask_cfgs: list[MLPOrResidualMaskConfig]):
        super().__init__()
        expanded_size = config.mlp_dims
        self.c_fc = nn.Linear(config.n_embd, expanded_size, bias=True)
        self.c_proj = nn.Linear(expanded_size, config.n_embd, bias=True)
        mask = torch.empty((len(mask_cfgs), expanded_size), requires_grad=False)
        for i, mask_cfg in enumerate(mask_cfgs):
            mask[i] = torch.full((expanded_size,), mask_cfg.neg_lr)
            mask[i][mask_cfg.dimensions] = mask_cfg.pos_lr
        ablate_mask = torch.ones((len(mask_cfgs), expanded_size), requires_grad=False)
        for i, mask_cfg in enumerate(mask_cfgs):
            ablate_mask[i][mask_cfg.dimensions] = 0.0
        self.register_buffer("mask", mask)
        self.register_buffer("ablate_mask", ablate_mask)

    def forward(self, x, tok_masks: Int[torch.Tensor, "batch seq"]):
        x = self.c_fc(x)
        x = F.gelu(x)

        if tok_masks is not None:
            mask = self.mask[tok_masks.long()]
            x = x * mask + (1.0 - mask) * x.detach()

        x = self.c_proj(x)
        return x, torch.tensor(0.0, device=x.device)

    def forward_ablated(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = x * self.ablate_mask[0]
        x = self.c_proj(x)
        return x

    def contract(self) -> RegularMLP:
        Config = namedtuple("Cfg", ["n_embd", "mlp_dims"])
        contracted_n_expanded = self.ablate_mask[0].sum().int()
        cfg = Config(self.c_fc.weight.size(1), contracted_n_expanded)
        new_mlp = RegularMLP(cfg)
        with torch.no_grad():
            new_mlp.c_fc.weight[:] = self.c_fc.weight[self.ablate_mask[0] == 1]
            new_mlp.c_fc.bias[:] = self.c_fc.bias[self.ablate_mask[0] == 1]
            new_mlp.c_proj.weight[:] = self.c_proj.weight[:, self.ablate_mask[0] == 1]
            new_mlp.c_proj.bias[:] = self.c_proj.bias

        return new_mlp


class ExpandedMLP(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        d_expanded,
        masking_is_param_level,
        expanded_dim_lr_forget: float,  # reasonable default: 1.0
        expanded_dim_lr_retain: float,  # reasonable default: 1.0 (if it is 0.0, no absorption happens)
        original_dim_lr_forget: float,  # reasonable default: 0.0
        original_dim_lr_retain: float,  # reasonable default: 1.0
    ):
        super().__init__()
        self.original_c_fc = RoutedLinear(d_model, d_ff, masking_is_param_level)
        self.expanded_c_fc = RoutedLinear(d_model, d_expanded, masking_is_param_level)
        self.expanded_dim_lr_forget = expanded_dim_lr_forget
        self.expanded_dim_lr_retain = expanded_dim_lr_retain
        self.original_dim_lr_forget = original_dim_lr_forget
        self.original_dim_lr_retain = original_dim_lr_retain
        self.c_proj = nn.Linear(d_ff + d_expanded, d_model)

    def forward(self, x, mask_ids):
        with torch.no_grad():
            if mask_ids is None:
                mask_ids = torch.zeros((x.size(0), x.size(1)), device=x.device)
            original_lrs = (
                1 - mask_ids
            ) * self.original_dim_lr_retain + mask_ids * self.original_dim_lr_forget
            expanded_lrs = (
                1 - mask_ids
            ) * self.expanded_dim_lr_retain + mask_ids * self.expanded_dim_lr_forget
        original = self.original_c_fc(x, original_lrs)
        expanded = self.expanded_c_fc(x, expanded_lrs)
        x = torch.cat([original, expanded], dim=-1)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x, torch.tensor(0, device=x.device)

    def forward_ablated(self, x):
        lrs = torch.ones((x.size(0), x.size(1)), device=x.device)
        original = self.original_c_fc(x, lrs)
        expanded = self.expanded_c_fc(x, lrs)
        # TODO optimize this
        expanded = torch.zeros_like(expanded)
        x = torch.cat([original, expanded], dim=-1)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

    def contract(self) -> RegularMLP:
        fc_weight = self.original_c_fc.weight[:]
        fc_bias = self.original_c_fc.bias[:]
        # TODO look into if slicing on right axis, pretty sure I am though
        proj_weight = self.c_proj.weight[:, : self.original_c_fc.out_features]
        proj_bias = self.c_proj.bias[:]

        regularMLP = RegularMLP(self.config)
        regularMLP.c_fc.weight[:] = fc_weight
        regularMLP.c_fc.bias[:] = fc_bias
        regularMLP.c_proj.weight[:] = proj_weight
        regularMLP.c_proj.bias[:] = proj_bias
        return regularMLP


if __name__ == "__main__":
    # setting up cuda
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    e = ExpandedMLP(1, 1, 1, False)
    with torch.no_grad():
        e.regular_c_fc.weight[0] = 1.0
        e.regular_c_fc.bias[0] = 0.0
        e.expanded_c_fc.weight[0] = 1.0
        e.expanded_c_fc.bias[0] = 0.0
        e.c_proj.weight[:] = torch.tensor([[1, 1]]).float()
        e.c_proj.bias[:] = torch.tensor([0]).float()
    x = torch.tensor([[1]], requires_grad=True, dtype=torch.float)
    y = e(x, torch.tensor([1])).sum()
    y.backward()
    print("grad on input", x.grad)
    print(e.regular_c_fc.weight.grad)
    print(e.regular_c_fc.bias.grad)
    print(e.expanded_c_fc.weight.grad)
    print(e.expanded_c_fc.bias.grad)
    print(e.c_proj.weight.grad)
    print(e.c_proj.bias.grad)


# %%
