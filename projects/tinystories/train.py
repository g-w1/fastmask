# %%
import argparse
import matplotlib.pyplot as plt
import math
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd
import torch
import tqdm
import transformers
import wandb
from dataclasses_json import dataclass_json
import builtins

from projects.shared.utils import get_gpu_with_most_memory, sanezip
from projects.shared.dataloader import DistributedDataLoader
from fm.model import (
    Block,
    CausalGroupedSelfAttention,
    Config,
    Embd,
    MLPOrResidualMaskConfig,
    RegularMLP,
    ExpandedMLP,
    RoutedTransformer,
    UnEmbd,
)
from projects.tinystories.shared import RunData, RunTypeConfig, get_type_of_mask

mask_type_prefix = "freq_based_masking"
np_type_of_mask, torch_type_of_mask = get_type_of_mask(mask_type_prefix)
gradient_accumulation_steps = 1
min_lr = 5e-5
lr = 5e-4
warmup_iters = 100
wandb_log = True
grad_clip = 1.0
batch_size = 80
block_size = 256
tokens_per_batch = batch_size * gradient_accumulation_steps * block_size


class RunType(Enum):
    erac_model = "erac_model"
    base_model = "base_model"
    pure_model = "pure_model"
    expanded_base_model = "expanded_base_model"
    rmu_model = "rmu_model"

    def __str__(self):
        return self.value

    def is_pure(self):
        return self == RunType.pure_model

    def is_rmu(self):
        return self == RunType.rmu_model

    def change_mask_type_prefix(self, old_prefix):
        if self == RunType.pure_model:
            return "pure"
        else:
            return old_prefix

    def get_run_type_config(self) -> RunTypeConfig:
        match self:
            case RunType.erac_model:
                return RunTypeConfig(
                    label="ERAC",
                    expand_model=True,
                    use_gradient_routing=True,
                    forget_data_labeling_percentage=1.0,
                    drop_labeled_forget_data=False,
                    drop_unlabeled_forget_data=False,
                    sort_forget_data_by_label=False,
                )
            case RunType.base_model:
                return RunTypeConfig(
                    label="base",
                    expand_model=False,
                    use_gradient_routing=False,
                    forget_data_labeling_percentage=1.0,
                    drop_labeled_forget_data=False,
                    drop_unlabeled_forget_data=False,
                    sort_forget_data_by_label=False,
                )
            case RunType.pure_model:
                return RunTypeConfig(
                    label="pure",
                    expand_model=False,
                    use_gradient_routing=False,
                    forget_data_labeling_percentage=1,
                    drop_labeled_forget_data=True,
                    drop_unlabeled_forget_data=False,
                    sort_forget_data_by_label=False,
                )
            case RunType.expanded_base_model:
                return RunTypeConfig(
                    label="expanded_base",
                    expand_model=True,
                    use_gradient_routing=False,
                    forget_data_labeling_percentage=1.0,
                    drop_labeled_forget_data=False,
                    drop_unlabeled_forget_data=False,
                    sort_forget_data_by_label=False,
                )
            case RunType.rmu_model:
                return RunTypeConfig(
                    label="rmu",
                    expand_model=False,
                    use_gradient_routing=False,
                    forget_data_labeling_percentage=1.0,
                    drop_labeled_forget_data=False,
                    drop_unlabeled_forget_data=False,
                    sort_forget_data_by_label=False,
                )


parser = argparse.ArgumentParser()
parser.add_argument("runs_id", type=str)
parser.add_argument("save_dir", type=str)
parser.add_argument("model_save_name", type=str)
parser.add_argument("run_type", type=RunType, choices=list(RunType))
parser.add_argument("--gpus_to_limit_to", type=int, nargs="*", help="GPUs to use")
parser.add_argument("--neg-lr", type=float, default=0)
parser.add_argument("--l1-coeff", type=float, default=0)
parser.add_argument("--dry_run", type=bool, default=False)
parser.add_argument("--compile", type=bool, default=True)
parser.add_argument("--do-retrain-evals", type=bool)
save_name = "param_level_3"
args = parser.parse_args(
    args=[
        "--neg-lr",
        "0.0",
        save_name,
        "param_level_masking",
        save_name,
        "erac_model",
        "--do-retrain-evals",
        "false",
    ]
)  # , "--dry_run", "True"])
dry_run = args.dry_run
runs_id = args.runs_id
model_save_name = args.model_save_name
gpus_to_limit_to: Optional[List[int]] = args.gpus_to_limit_to
run_type = args.run_type
do_retrain_evals = args.do_retrain_evals

mask_type_prefix = run_type.change_mask_type_prefix(mask_type_prefix)
run_type_config = run_type.get_run_type_config()


print("tokens per batch:", tokens_per_batch)

max_iters = int(
    400_000
    * (
        1 if not run_type.is_pure() else 0.8
    )  # we want to train for less steps if we are doing pure masking
    * block_size
    // tokens_per_batch
)
if dry_run:
    max_iters = 100
lr_decay_iters = max_iters - warmup_iters


# turn off lr scheduling to simplify things
def get_lr(it):
    return lr


# setting up cuda
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)  # type: ignore
scaler = torch.amp.GradScaler()  # type: ignore


device = get_gpu_with_most_memory(gpus_to_limit_to)
# allocate some memory so it doesn't get allocated later if we are doing bulk runs
dummy_tensor = torch.zeros((100, 100), dtype=ptdtype, device=device)
torch.cuda.set_device(device)
seed_offset = 0


# MODEL SETUP STARTS HERE
cfg = Config(
    block_size=block_size,
    n_layer=8,
    n_head=16,
    n_key_value_head=16,
    tie_weights=False,
    n_embd=512,
    mlp_dims=512 * 4 + 64 if run_type_config.expand_model else 512 * 4,
)

if run_type_config.expand_model:
    l1_coeff = args.l1_coeff
    neg_lr = args.neg_lr if run_type_config.use_gradient_routing else 1.0
    print("USING neg_lr:", neg_lr)
    print("USING l1_coeff:", l1_coeff)

    mlps = [
        ExpandedMLP(
            cfg,
            64,
            masking_is_param_level=True,
            expanded_dim_lr_forget=1.0,
            expanded_dim_lr_retain=1.0,
            original_dim_lr_forget=neg_lr,
            original_dim_lr_retain=1.0,
        )
        for _ in range(cfg.n_layer)
    ]
    attns = [CausalGroupedSelfAttention(cfg) for _ in range(cfg.n_layer)]
    blocks = [Block(cfg, attn, mlp) for attn, mlp in sanezip(attns, mlps)]
    embd = Embd(cfg)
    unembd = UnEmbd(cfg)
    model: torch.nn.Module = RoutedTransformer(cfg, embd, unembd, blocks)
else:
    mlps = [
        RegularMLP(cfg)  # we do this just so we can contract for the expanded_base case
        for _ in range(cfg.n_layer)
    ]
    attns = [CausalGroupedSelfAttention(cfg) for _ in range(cfg.n_layer)]
    blocks = [Block(cfg, attn, mlp) for attn, mlp in sanezip(attns, mlps)]
    embd = Embd(cfg)
    unembd = UnEmbd(cfg)
    if cfg.tie_weights:
        unembd.unembedding.weight = embd.embedding.weight
    model: torch.nn.Module = RoutedTransformer(cfg, embd, unembd, blocks)
optim = model.configure_optimizers(0.1, lr, (0.9, 0.95), device)
model.to(device)


# MODEL SETUP ENDS HERE

# compile the model
if args.compile:
    print("compiling the model... (takes a ~minute)")
    model: torch.nn.Module = torch.compile(model)  # type: ignore

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")


train_dataloader = DistributedDataLoader(
    f"{mask_type_prefix}_tinystories_train.bin",
    batch_size,
    cfg.block_size,
    0,
    1,
    mask_ids_filename=f"{mask_type_prefix}_tinystories_train_masks.bin",
    types_of_mask=(np_type_of_mask, torch_type_of_mask),
)
val_forget_dataloader = DistributedDataLoader(
    f"{mask_type_prefix}_tinystories_only_forget_validation.bin",
    batch_size,
    cfg.block_size,
    0,
    1,
    mask_ids_filename=f"{mask_type_prefix}_tinystories_only_forget_validation_masks.bin",
    types_of_mask=(np_type_of_mask, torch_type_of_mask),
)  # we only ever evaluate val on the master process, so only need one parallel copy
val_retain_dataloader = DistributedDataLoader(
    f"{mask_type_prefix}_tinystories_only_retain_validation.bin",
    batch_size,
    cfg.block_size,
    0,
    1,
    mask_ids_filename=f"{mask_type_prefix}_tinystories_only_retain_validation_masks.bin",
    types_of_mask=(np_type_of_mask, torch_type_of_mask),
)  # we only ever evaluate val on the master process, so only need one parallel copy
train_forget_dataloader = DistributedDataLoader(
    f"{mask_type_prefix}_tinystories_only_forget_train.bin",
    batch_size,
    cfg.block_size,
    0,
    1,
    mask_ids_filename=f"{mask_type_prefix}_tinystories_only_forget_train_masks.bin",
    types_of_mask=(np_type_of_mask, torch_type_of_mask),
)  # we only use this on the master process
train_retain_dataloader = DistributedDataLoader(
    f"{mask_type_prefix}_tinystories_only_retain_train.bin",
    batch_size,
    cfg.block_size,
    0,
    1,
    mask_ids_filename=f"{mask_type_prefix}_tinystories_only_retain_train_masks.bin",
    types_of_mask=(np_type_of_mask, torch_type_of_mask),
)


def get_dataloader(split):
    if split == "train":
        return train_dataloader
    elif split == "train_forget":
        return train_forget_dataloader
    elif split == "train_retain":
        return train_retain_dataloader
    elif split == "val_forget":
        return val_forget_dataloader
    elif split == "val_retain":
        return val_retain_dataloader
    else:
        raise ValueError("invalid split")


def get_batch(split):
    dataloader = get_dataloader(split)

    x, y, mask_ids = dataloader.next_batch()  # type: ignore

    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y, mask_ids = (
        x.pin_memory().to(device, non_blocking=True),
        y.pin_memory().to(device, non_blocking=True),
        mask_ids.pin_memory().to(device, non_blocking=True),
    )
    return x, y, mask_ids


@torch.no_grad()
def eval_on_val(use_retain: bool, model, ablate):
    model.eval()
    dataloader_str = "val_retain" if use_retain else "val_forget"
    dataloader = get_dataloader(dataloader_str)
    dataloader.reset()  # always eval on the same items
    val_loss = 0.0
    steps = 50
    for _ in range(steps):
        x, y, mask_ids = get_batch(dataloader_str)
        with ctx:
            if ablate:
                _, lm_loss = model.forward_ablated(x, y)
            else:
                logits, aux_loss, lm_loss = model(x, y, mask_ids)
            loss = lm_loss
            val_loss += loss.item()
    val_loss /= steps
    model.train()
    return val_loss


if wandb_log:
    import wandb

    project_dir = os.path.dirname(os.path.abspath(__file__))
    wandb.init(
        project="fastmask-tinystories",
        name="fastmask-tinystories-1",
        config={
            "model_cfg": cfg,
            "runs_id": runs_id,
            "model_save_name": model_save_name,
        },
        settings=wandb.Settings(code_dir=project_dir),
        dir=project_dir,
    )
    wandb.run.log_code(  # type: ignore
        project_dir,
    )
mlp_l1_norm_over_training = []

del dummy_tensor  # free up allocated tensor
# train loop
X, Y, tok_masks = get_batch("train")  # fetch the very first batch
for iter_num in (pbar := tqdm.trange(max_iters)):
    iter_lr = get_lr(iter_num)
    for param_group in optim.param_groups:
        param_group["lr"] = iter_lr
    los = 0.0
    aux_los = 0.0
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, aux_loss, lm_loss = model(X, Y, tok_masks)
            loss = lm_loss + aux_loss
            loss /= gradient_accumulation_steps
        los += lm_loss.item() / gradient_accumulation_steps
        aux_los += aux_loss.item() / gradient_accumulation_steps

        do_residual_coherence = True
        if do_residual_coherence:
            X, Y, _ = get_batch("train_retain")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        if do_residual_coherence:
            with ctx:
                _, _, residual_coherence_loss = model(X, Y, torch.ones_like(Y))
                residual_coherence_loss /= gradient_accumulation_steps
            if wandb_log:
                wandb.log(
                    {"residual_coherence_loss": residual_coherence_loss.item()},
                    step=iter_num,
                )
        X, Y, tok_masks = get_batch("train")  # prefetch the next batch async
        if do_residual_coherence:
            scaler.scale(residual_coherence_loss).backward()
    if wandb_log:
        wandb.log(
            {
                "loss_on_master_process_train": los,
                "aux_loss_on_master_process_train": aux_los,
            },
            step=iter_num,
        )
    pbar.set_postfix({"loss": los})
    if grad_clip != 0.0:
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optim)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optim.zero_grad(set_to_none=True)
    mlp_l1_norm_sum = sum(
        block.mlp.magnitude_of_proj_up_matrix() for block in model.transformer.blocks
    )
    wandb.log({"l1_norm_mlp_proj_matrix": mlp_l1_norm_sum}, step=iter_num)
    mlp_l1_norm_over_training.append(mlp_l1_norm_sum.item())

    if iter_num % 250 == 0 or iter_num == max_iters - 1:
        val_loss_forget_ablated = eval_on_val(False, model, ablate=True)
        val_loss_retain_ablated = eval_on_val(True, model, ablate=True)
        val_loss_forget = eval_on_val(False, model, ablate=False)
        val_loss_retain = eval_on_val(True, model, ablate=False)
        if wandb_log:
            wandb.log(
                {
                    "val_loss_forget": val_loss_forget,
                    "val_loss_retain": val_loss_retain,
                    "val_loss_forget_ablated": val_loss_forget_ablated,
                    "val_loss_retain_ablated": val_loss_retain_ablated,
                },
                step=iter_num,
            )
retain_loss_before_contract = val_loss_retain  # type: ignore
forget_loss_before_contract = val_loss_forget  # type: ignore

if run_type_config.expand_model:
    print("DOING COHERENCE; contracting model")
    contracted_model: torch.nn.Module = model.contract()  # type: ignore
    contracted_model.to(device)
    contracted_model: torch.nn.Module = torch.compile(contracted_model)  # type: ignore
    val_loss_forget = eval_on_val(False, contracted_model, ablate=False)
    val_loss_retain = eval_on_val(True, contracted_model, ablate=False)
    retain_loss_after_contract = val_loss_retain  # type: ignore
    forget_loss_after_contract = val_loss_forget  # type: ignore
    if wandb_log:
        wandb.log(
            {
                "val_loss_forget": val_loss_forget,
                "val_loss_retain": val_loss_retain,
            },
        )
    del model, optim
    new_optim = contracted_model.configure_optimizers(0.1, 5e-5, (0.9, 0.95), device)
    X, Y, tok_masks = get_batch("train_retain")
    best_coherence_loss = float("inf")
    coherence_steps = 10
    best_coherence_state_dict = deepcopy(contracted_model.state_dict())
    for iter_num in tqdm.trange(coherence_steps):
        los = 0.0
        with ctx:
            logits, aux_loss, lm_loss = contracted_model(X, Y, tok_masks)
            loss = lm_loss + aux_loss
        los = lm_loss.item()
        scaler.scale(loss).backward()
        if wandb_log:
            wandb.log({"loss_on_master_process_train": los})
        if grad_clip != 0.0:
            scaler.unscale_(new_optim)
            torch.nn.utils.clip_grad_norm_(contracted_model.parameters(), grad_clip)
        scaler.step(new_optim)
        scaler.update()
        new_optim.zero_grad(set_to_none=True)
        val_loss_retain = eval_on_val(True, contracted_model, ablate=False)
        if val_loss_retain < best_coherence_loss:
            best_coherence_loss = val_loss_retain
            best_coherence_state_dict = deepcopy(contracted_model.state_dict())

    contracted_model.load_state_dict(best_coherence_state_dict)  # type: ignore
else:
    contracted_model = model
    best_coherence_state_dict = deepcopy(model.state_dict())
    forget_loss_after_contract = forget_loss_before_contract
    retain_loss_after_contract = retain_loss_before_contract
if run_type.is_rmu():
    frozen_model = deepcopy(contracted_model)
    frozen_model.to(device)
    layers_to_train = [0, 1, 2, 3, 4, 5]
    rmu_steps = 500
    steering_coef = 100
    retain_weight = 200  # rmu alpha
    lr = 5e-4
    params = []
    for block_i in layers_to_train:
        params.append(contracted_model.transformer.blocks[block_i].mlp.c_proj.weight)
    new_optim = torch.optim.AdamW(params, lr=lr)
    random_vec = torch.rand(cfg.n_embd, device=device)
    control_vec = random_vec / random_vec.norm() * steering_coef

    for iter_num in (pbar := tqdm.trange(rmu_steps)):
        forget_X, forget_Y, forget_tok_masks = get_batch("train_forget")
        retain_X, retain_Y, retain_tok_masks = get_batch("train_retain")
        # forget
        forget_activations, _ = contracted_model(
            forget_X, forget_Y, forget_tok_masks, stop_at_layer=max(layers_to_train)
        )
        forget_loss = torch.nn.functional.mse_loss(
            forget_activations, control_vec[None, None, :]
        )
        # retain
        retain_activations, _ = contracted_model(
            retain_X, retain_Y, retain_tok_masks, stop_at_layer=max(layers_to_train)
        )
        with torch.no_grad():
            frozen_retain_activations, _ = frozen_model(
                retain_X, retain_Y, retain_tok_masks, stop_at_layer=max(layers_to_train)
            )
        retain_mse = torch.nn.functional.mse_loss(
            retain_activations, frozen_retain_activations
        )
        retain_loss = retain_mse * retain_weight
        loss = forget_loss + retain_loss
        loss.backward()
        if wandb_log:
            wandb.log(
                {
                    "rmu/forget_loss": forget_loss.item(),
                    "rmu/retain_loss": retain_loss.item(),
                },
            )
        new_optim.step()
        new_optim.zero_grad(set_to_none=True)
        pbar.set_postfix(
            {"forget_loss": forget_loss.item(), "retain_loss": retain_loss.item()}
        )
    best_coherence_state_dict = deepcopy(contracted_model.state_dict())
    contracted_model.load_state_dict(best_coherence_state_dict)  # type: ignore


val_loss_forget = eval_on_val(False, contracted_model, ablate=False)
val_loss_retain = eval_on_val(True, contracted_model, ablate=False)

if wandb_log:
    wandb.log(
        {
            "val_loss_forget": val_loss_forget,
            "val_loss_retain": val_loss_retain,
        },
    )

if do_retrain_evals:
    min_retrain_forget_loss = float("inf")
    min_retrain_retain_loss = float("inf")
    print("Doing retrain on val forget eval")
    forget_retrain_losses_outer = []
    retain_retrain_losses_outer = []
    num_stories_retrain = [64]
    for num_seqs in num_stories_retrain:
        forget_retrain_losses_inner = []
        retain_retrain_losses_inner = []
        print(f"retraining on {num_seqs} stories")
        new_optim = contracted_model.configure_optimizers(
            0.1, 5e-5, (0.9, 0.95), device
        )  # reset the optimizer
        contracted_model.load_state_dict(best_coherence_state_dict)
        val_forget_loss = eval_on_val(False, contracted_model, ablate=False)
        val_retain_loss = eval_on_val(True, contracted_model, ablate=False)
        forget_retrain_losses_inner.append(val_forget_loss)
        retain_retrain_losses_inner.append(val_retain_loss)
        if val_forget_loss < min_retrain_forget_loss:
            min_retrain_forget_loss = val_forget_loss
        if val_retain_loss < min_retrain_retain_loss:
            min_retrain_retain_loss = val_retain_loss
        if wandb_log:
            wandb.log(
                {
                    f"retrain_evals_{num_seqs}/forget": val_forget_loss,
                    f"retrain_evals_{num_seqs}/retain": val_retain_loss,
                },
            )
        forget_dataloader = DistributedDataLoader(
            f"{mask_type_prefix}_tinystories_only_forget_train.bin",
            num_seqs,
            cfg.block_size,
            0,
            1,
            mask_ids_filename=f"{mask_type_prefix}_tinystories_only_forget_train_masks.bin",
            types_of_mask=(np_type_of_mask, torch_type_of_mask),
        )  # we only ever evaluate val on the master process, so only need one parallel copy
        x, y, mask_ids = forget_dataloader.next_batch()  # type: ignore
        x, y, mask_ids = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
            mask_ids.pin_memory().to(device, non_blocking=True),
        )
        for steps in range(20):
            los = 0.0
            with ctx:
                logits, aux_loss, lm_loss = contracted_model(x, y, mask_ids)
                loss = lm_loss + aux_loss
            los = lm_loss.item()
            scaler.scale(loss).backward()
            if grad_clip != 0.0:
                scaler.unscale_(new_optim)
                torch.nn.utils.clip_grad_norm_(contracted_model.parameters(), grad_clip)
            scaler.step(new_optim)
            scaler.update()
            new_optim.zero_grad(set_to_none=True)

            val_forget_loss = eval_on_val(False, contracted_model, ablate=False)
            val_retain_loss = eval_on_val(True, contracted_model, ablate=False)
            forget_retrain_losses_inner.append(val_forget_loss)
            retain_retrain_losses_inner.append(val_retain_loss)
            if wandb_log:
                wandb.log(
                    {
                        f"retrain_evals_{num_seqs}/forget": val_forget_loss,
                        f"retrain_evals_{num_seqs}/retain": val_retain_loss,
                    },
                )
        forget_retrain_losses_outer.append(forget_retrain_losses_inner)
        retain_retrain_losses_outer.append(retain_retrain_losses_inner)
else:
    forget_retrain_losses_outer = []
    retain_retrain_losses_outer = []
    num_stories_retrain = []
    min_retrain_forget_loss = float("inf")
    min_retrain_retain_loss = float("inf")

run_data = RunData(
    forget_loss_before_contract=forget_loss_before_contract,
    retain_loss_before_contract=retain_loss_before_contract,
    forget_loss_after_contract=forget_loss_after_contract,
    retain_loss_after_contract=retain_loss_after_contract,
    forget_retrain_losses=forget_retrain_losses_outer,
    retain_retrain_losses=retain_retrain_losses_outer,
    num_stories_retrain=num_stories_retrain,
    metadata=run_type_config,
    model_save_name=model_save_name,
)

run_type_str = run_type_config.label
save_path = os.path.join(args.save_dir, f"{model_save_name}.json")
json_str: str = run_data.to_json()  # type: ignore
os.makedirs(args.save_dir, exist_ok=True)
with open(save_path, "w") as f:
    f.write(json_str)

fig, ax = plt.subplots()
ax.plot(mlp_l1_norm_over_training)
ax.set_xlabel("iteration")
ax.set_ylabel("sum of l1 norm of MLP projection weight matrix + bias over all layers")
plt.show()
# %%
