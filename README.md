# fastmask

Fast and flexible gradient routing for transformers.

## How to use?

Fastmask aims to implement a flexible interface for specifying which allows specifying **what data** gets routed to **where in the transformer**.

The core object is a `RoutedTransformer`, which is built from dependency-injected blocks.

In a forward pass, if `model` is a `RoutedTransformer` and the input is of shape `[batch, seq, d_model]`, you should also have a `mask_ids` tensor of shape `[seq, d_model]` with each entry in it specifying which class that token belongs to. In our previous work, we've adopted the convention of `0` marking a token as being in the "forget" set and `1` marking "retain." Here is an example of how one might build a `RoutedTransformer`:
```python3
mlps = [
    ExpandedMLP(
        cfg,
        d_expand=64,
        masking_is_param_level=True,
        expanded_dim_lr_forget=1.0,
        expanded_dim_lr_retain=1.0,
        original_dim_lr_forget=0.0,
        original_dim_lr_retain=1.0,
    )
    if i < 9
    else RegularMLP(cfg)
    for i in range(n_layers)
]
attns = [CausalGroupedSelfAttention(cfg) for _ in range(n_layers)]
blocks = [Block(cfg, attn, mlp) for attn, mlp in zip(attns, mlps)]
embd = Embd(cfg)
unembd = UnEmbd(cfg)
model = RoutedTransformer(cfg, embd, unembd, blocks)
```
Then, to train the model, first create the `mask_ids` (using whatever masking rule you want) and then pass it to the model.
```
mask_ids = rule(Y)
logits, aux_loss, lm_loss = model(X, Y, mask_ids)
```
## How to build modules

If you want to experiment with different types of routing, it should be pretty simple. Just create a new MLP/attention module with this structure:
```python3
class CustomMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # set up the module

    def forward(self, x, mask_ids: Float[torch.Tensor, "batch seq"]):
        # do some masking, detach some stuff, use some custom operations

        # return both the output and an auxillary loss scalar tensor
        return x, torch.tensor(0.0, device=x.device)

    def forward_ablated(self, x):
        # do a forward pass with the "forget" dimensions ablated
        return x

    def contract(self) -> RegularMLP:
        # create a new Module from this one with the expanded dimensions ablated away
        return new_mlp
```

If you want some inspiration, see `DiscreteMaskingMLP` (for when `mask_ids` is either `0` or `1` and you want to mask on MLP activations), `ContiniousMaskingMLP` (for when `mask_ids` can contain values between `0` and `1` and you want to mask on MLP activations), or `ExpandedMLP` for a more general implementation that allows masking MLP activations *or* just the gradients on the weights.

## A few nice things

- You can sample some text with `model.generate(`.
- You can get a new "contracted" model with `model.contract(` (which recursively calls `contract` on all of the modules passed in).
- You should be able to compile the model to make it run faster with `torch.compile(`.

## Example
See `train_gpt2.py` for an example training loop with autocast, gradient accumulation, and multi-gpu support (thanks Karpathy!).

## Optimizations

I took a bunch of inspiration from Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), but then incorporated some architecture changes from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/). I didn't take all the optimizations (e.g. it still uses AdamW) so I suspect there is still some low hanging fruit. If you see that something could be optimized, feel free to let me know / change it.

## Philosophy

- Should be as simple as possible while still having a nice interface.
- Technical limitations shouldn't constrain masking schemes.
- Should be modular