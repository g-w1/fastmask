from typing import Callable, Tuple
import tqdm
import torch


@torch.no_grad()
def eval_on_val(dataloader, model, ablate=False):
    model.eval()
    dataloader.reset()  # always eval on the same items
    val_loss = 0.0
    steps = 50
    device = next(model.parameters()).device
    for _ in range(steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        if ablate:
            _, lm_loss = model.forward_ablated(x, y)
        else:
            _, _, lm_loss = model(x, y)
        loss = lm_loss
        val_loss += loss.item()
    val_loss /= steps
    model.train()
    return val_loss


def load_special_batch(
    text_path: str, tokenizer, block_len: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    torch_tokens = torch.tensor(tokens, dtype=torch.long)
    x = torch.stack(
        [
            torch_tokens[i : i + block_len]
            for i in range(0, len(torch_tokens) - block_len, block_len)
        ]
    )
    y = torch.stack(
        [
            torch_tokens[i + 1 : i + 1 + block_len]
            for i in range(0, len(torch_tokens) - block_len, block_len)
        ]
    )
    x, y = (
        x.pin_memory().to(device, non_blocking=True),
        y.pin_memory().to(device, non_blocking=True),
    )
    return x, y


def eval_on_special_batch(
    model,
    text_path: str,
    batch_size: int,
    block_len: int,
    token_masking_rule: Callable,
    device,
    tokenizer,
    include_masked_tokens,
    ablate=False,
) -> float:
    x, y = load_special_batch(text_path, tokenizer, block_len, device)
    total_loss = 0
    num_batches = 0
    with torch.inference_mode():
        for i in range(0, len(x), batch_size):
            batch_x = x[i : i + batch_size]
            batch_y = y[i : i + batch_size]
            mask = token_masking_rule(batch_y)
            if ablate:
                _, loss = model.forward_ablated(batch_x, batch_y, reduce_loss=False)
            else:
                _, _, loss = model(batch_x, batch_y, mask, reduce_loss=False)
            if not include_masked_tokens:
                loss = loss * mask.flatten(0, 1)
                loss = loss.mean()
            else:
                loss = loss.mean()
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches


@torch.inference_mode()
def estimate_custom_loss(
    custom_path: str,
    model,
    tokenizer,
    token_masking_rule,
    include_masked_tokens,
    ablate=False,
):
    model.eval()
    loss = eval_on_special_batch(
        model,
        custom_path,
        100,
        1024,
        token_masking_rule,
        next(model.parameters()).device,
        tokenizer,
        include_masked_tokens=include_masked_tokens,
        ablate=ablate,
    )
    model.train()
    return loss
