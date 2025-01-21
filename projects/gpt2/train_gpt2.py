# %%
import math
import torch
import os
import datetime
import tqdm
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from dotenv import load_dotenv
from fm.model import (
    Config,
    ExpandedMLP,
    MLPOrResidualMaskConfig,
    DiscreteMaskingMLP,
    RegularMLP,
    CausalGroupedSelfAttention,
    Block,
    RoutedTransformer,
    Embd,
    UnEmbd,
)
import wandb
from dataloader import DistributedDataLoader
import evals

gradient_accumulation_steps = 48
lr = 1e-3
cooldown_frac = 0.4
wandb_log = True
grad_clip = 1.0
batch_size = 6
block_size = 1024
tokens_per_batch = batch_size * gradient_accumulation_steps * block_size
print("tokens per batch:", tokens_per_batch)
ddp_backend = "nccl"

tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(
        backend=ddp_backend, timeout=datetime.timedelta(minutes=120)
    )  # the master process will be doing evals that take a while so we give a lot of time
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    device = torch.device("cuda")
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_local_rank = 0


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if master_process:
        print(*args, **kwargs)


n_dims_expand = 64
d_model = 768
n_layers = 12
n_head = 6
cfg = Config(
    block_size=block_size,
    vocab_size=tokenizer.vocab_size,
    n_layer=n_layers,
    n_head=n_head,
    n_key_value_head=6,
    tie_weights=False,
    n_embd=d_model,
    mlp_dims=d_model * 4,
    expand_dims=n_dims_expand,
)


max_iters = int(
    10_000_000_000  # 10B tokens for now
    // tokens_per_batch
)


# TODO replace lr and lr decay with modded nanogpt's
def get_lr(it):
    t = 1 - it / max_iters  # time remaining in training
    if t > cooldown_frac:
        return lr
    else:
        return lr * (t / cooldown_frac)


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

tokens_to_mask = {
    "COVID": 0,
    " COVID": 0,
    "RNA": 0,
    " infections": 0,
    "DNA": 0,
    " genome": 0,
    " virus": 0,
    " gene": 0,
    " viruses": 0,
    " mutations": 0,
    " antibodies": 0,
    " influenza": 0,
    " bacteria": 0,
    "PCR": 0,
    " cell": 0,
    " herpes": 0,
    " bacterial": 0,
    " pathogens": 0,
    " tumor": 0,
    " vaccine": 0,
}
token_to_label = torch.ones(cfg.vocab_size, device=device)
for token, label in tokens_to_mask.items():
    token_id = tokenizer.encode(token)[0]
    token_to_label[token_id] = label


def rule(ys):
    return token_to_label[ys]


# DEFINE THE MODEL
mlps = [
    # RegularMLP(cfg)
    ExpandedMLP(
        cfg,
        n_dims_expand,
        masking_is_param_level=False,
        expanded_dim_lr_forget=1.0,
        expanded_dim_lr_retain=1.0,
        original_dim_lr_forget=0.0,
        original_dim_lr_retain=1.0,
    )
    # DiscreteMaskingMLP(
    #    cfg,
    #    [
    #        MLPOrResidualMaskConfig(list(range(64)), 1.0, 0),
    #        MLPOrResidualMaskConfig(list(range(d_model)), 1.0, 1.0),
    #    ],
    # )
    if i < 6
    else RegularMLP(cfg)
    for i in range(n_layers)
]
attns = [CausalGroupedSelfAttention(cfg) for _ in range(n_layers)]
blocks = [Block(cfg, attn, mlp) for attn, mlp in zip(attns, mlps)]
embd = Embd(cfg)
unembd = UnEmbd(cfg)
if cfg.tie_weights:
    unembd.unembedding.weight = embd.embedding.weight
model = RoutedTransformer(cfg, embd, unembd, blocks)
model.to(device)


print("compiling the model... (takes a ~minute)")
# I need to set to 128 for ExpandedMLP, since it recompiles it a bunch
torch._dynamo.config.cache_size_limit = 128

model: torch.nn.Module = torch.compile(model)  # type: ignore
if ddp:
    ddp_model = DDP(
        model,
        device_ids=[ddp_local_rank],
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
else:
    ddp_model = model
base_model = ddp_model.module if ddp else model

train_dataloader = DistributedDataLoader(
    "fineweb_10BT_train_wmdp_0.5.bin",
    batch_size,
    block_size,
    process_rank=ddp_local_rank,
    num_processes=ddp_world_size,
)
pure_train_dataloader = DistributedDataLoader(
    "fineweb_10BT_residual_coherence_set_wmdp_0.5.bin",
    batch_size,
    block_size,
    process_rank=ddp_local_rank,
    num_processes=ddp_world_size,
)
val_dataloader = DistributedDataLoader(
    "fineweb_10BT_val_wmdp_0.5.bin",
    batch_size,
    block_size,
    process_rank=ddp_local_rank,
    num_processes=ddp_world_size,
)


optim = model.configure_optimizers(0.1, lr, (0.9, 0.95), device)

if master_process and wandb_log:
    load_dotenv()
    api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=api_key)
    wandb.init(project="fastmask", name="fastmask")
    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
else:
    wandb_log = False


X, Y = train_dataloader.next_batch()
X, Y = (X.pin_memory().to(device), Y.pin_memory().to(device))
if master_process:
    pbar = tqdm.trange(max_iters)
else:
    pbar = range(max_iters)
mask_ids = torch.ones_like(rule(Y))
for iter_num in pbar:
    iter_lr = get_lr(iter_num)
    for param_group in optim.param_groups:
        param_group["lr"] = iter_lr
    los = 0.0
    aux_los = 0.0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            ddp_model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, aux_loss, lm_loss = ddp_model(X, Y, mask_ids)
            loss = lm_loss + aux_loss
            loss /= gradient_accumulation_steps
        los += lm_loss.item() / gradient_accumulation_steps
        aux_los += aux_loss.item() / gradient_accumulation_steps

        do_residual_coherence = iter_num % 200 == 0  # do it on 0.5% of steps

        if do_residual_coherence:
            X, Y = pure_train_dataloader.next_batch()
            X, Y = (
                X.pin_memory().to(device, non_blocking=True),
                Y.pin_memory().to(device, non_blocking=True),
            )
        scaler.scale(loss).backward()
        # residual coherence step
        if do_residual_coherence:
            with ctx:
                _, residual_coherence_loss = base_model.forward_ablated(X, Y)
                residual_coherence_loss /= gradient_accumulation_steps

        X, Y = train_dataloader.next_batch()
        X, Y = (
            X.pin_memory().to(device, non_blocking=True),
            Y.pin_memory().to(device, non_blocking=True),
        )
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
    if master_process:
        pbar.set_postfix({"loss": los})
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optim)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optim.zero_grad(set_to_none=True)

    if iter_num % 250 == 0 or iter_num == max_iters - 1:
        with ctx:
            val_loss_forget = evals.estimate_custom_loss(
                "virology.txt", model, tokenizer, rule, True, ablate=False
            )
            val_loss_forget_ablated = evals.estimate_custom_loss(
                "virology.txt", model, tokenizer, rule, True, ablate=True
            )
            val_loss_retain = evals.eval_on_val(val_dataloader, model, ablate=False)
            val_loss_retain_ablated = evals.eval_on_val(
                val_dataloader, model, ablate=True
            )
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
wandb.finish()

# %%
