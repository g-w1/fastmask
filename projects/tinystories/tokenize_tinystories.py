from collections import Counter
import os
import re
from typing import List, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import torch as t
from projects.tinystories.shared import get_type_of_mask


# a bunch of this code is from the original gradient routing code
def get_token_frequencies(
    stories: list[str],
    tokenizer,
    synthetic_token_ct: Union[int, float],
    truncate_at: int | None,
):
    counter = Counter()
    for story in tqdm(stories):
        if truncate_at:
            tokens = tokenizer(
                story, max_length=truncate_at, add_special_tokens=False, truncation=True
            )["input_ids"]
        else:
            tokens = tokenizer(story, add_special_tokens=False)["input_ids"]
        counter.update(tokens)

    counts = t.Tensor([counter[tok_idx] for tok_idx in range(tokenizer.vocab_size)])
    counts = counts + synthetic_token_ct
    freq = counts / counts.sum()
    return freq


def get_token_freq_masking_rule(
    retain_stories: list[str],
    forget_stories: list[str],
    num_stories: int,
    truncate_at: int | None,
    num_synthetic_tokens_retain: Union[int, float],  # encodes uniform prior over toks
    num_synthetic_tokens_forget: Union[int, float],  # encodes uniform prior over toks
    scale: float,
    bias: float,
    tokenizer,
    device: t.device,
):
    print("Getting token frequencies...")
    retain_freq = get_token_frequencies(
        retain_stories[:num_stories],
        tokenizer,
        num_synthetic_tokens_retain,
        truncate_at,
    )
    retain_counts_posterior = retain_freq
    forget_freq = get_token_frequencies(
        forget_stories[:num_stories],
        tokenizer,
        num_synthetic_tokens_forget,
        truncate_at,
    )
    forget_counts_posterior = forget_freq

    ratio = forget_counts_posterior / retain_counts_posterior
    inverse_ratio = retain_counts_posterior / forget_counts_posterior
    ratio_diff = ratio - inverse_ratio
    mask_weight = 1 - t.nn.functional.sigmoid(scale * ratio_diff + bias).to(device)

    info = dict(
        retain_freq=retain_freq.cpu().numpy(),
        forget_freq=forget_freq.cpu().numpy(),
        ratio=ratio.cpu().numpy(),
        inverse_ratio=inverse_ratio.cpu().numpy(),
        ratio_diff=ratio_diff.cpu().numpy(),
        mask_weight=mask_weight.cpu().numpy(),
    )

    def token_freq_masking_rule(toks):
        left_shifted_toks = toks[:, 1:]
        return mask_weight[t.clamp(left_shifted_toks, max=tokenizer.vocab_size - 1)]

    return token_freq_masking_rule, info


def target_word_snippets(
    story: str, target_keywords: List[str], snippet_len: int = 20
) -> List[str]:
    """
    Returns a list of snippets of length `snippet_len` containing the target words.
    """
    snippets = []
    for keyword in target_keywords:
        escaped_substring = re.escape(keyword)
        # Create a pattern that includes the substring and surrounding context
        pattern = f"(.{{0,{snippet_len}}}){escaped_substring}(.{{0,{snippet_len}}})"
        matches = re.finditer(pattern, story, re.IGNORECASE)
        for match in matches:
            full_match = match.group()
            snippets.append(full_match)
    return snippets


def split_stories_by_concept(
    stories: List[str],
    target_words,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    for w in target_words:
        assert not w.startswith(" ") and not w.endswith(" "), (
            "We filter stories by words, not tokens, so target words should not have leading or trailing spaces."
            "This used to be a warning, but we ignored it, so it's an assert now."
        )
    target_words = [w.strip() for w in target_words]
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(word) for word in target_words) + r")\b",
        re.IGNORECASE,
    )

    concept_stories = []
    other_stories = []
    for story_idx, story in tqdm(enumerate(stories), total=len(stories)):
        assert isinstance(story, str), f"`story` should be a string, got {type(story)}"
        if pattern.search(story):
            if verbose:
                print(f"\nStory {story_idx}")
                for snippet in target_word_snippets(story, target_words):
                    print(f'"{snippet}"')
            concept_stories.append(story)
        else:
            other_stories.append(story)
    return concept_stories, other_stories


def split_and_label_stories_by_concept(
    stories: List[str],
    target_words,
    verbose: bool = False,
) -> Tuple[List[tuple], List[tuple]]:
    concept, other = split_stories_by_concept(stories, target_words, verbose=verbose)
    return [(concept, 0) for concept in concept], [(other, 1) for other in other]


# number of workers in .map() call
num_proc = 12
num_proc_load_dataset = num_proc

enc = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
# prefix = "full_seq_masking"
prefix = "freq_based_masking"
# prefix = "pure"

if __name__ == "__main__":
    np_type_of_mask, torch_type_of_mask = get_type_of_mask(prefix)
    dataset = load_dataset(
        "delphi-suite/stories",
        num_proc=num_proc_load_dataset,
    )

    all_stories = list(dataset["train"]["story"]) + list(dataset["validation"]["story"])  # type: ignore

    words_to_localize = [
        "tree",
        "trees",
        "forest",
        "forests",
        "woodland",
        "woodlands",
    ]
    forget_stories, retain_stories = split_and_label_stories_by_concept(
        all_stories, words_to_localize
    )

    # Create separate datasets for forget and retain stories
    forget_text, forget_labels = zip(*forget_stories)
    retain_text, retain_labels = zip(*retain_stories)

    forget_data_dict = {
        "text": forget_text,
        "label": forget_labels,
    }
    retain_data_dict = {
        "text": retain_text,
        "label": retain_labels,
    }

    forget_dataset = Dataset.from_dict(forget_data_dict)
    retain_dataset = Dataset.from_dict(retain_data_dict)

    # Set up token frequency masking
    num_token_freq_calculate_stories = 25000
    token_freq_kwargs = dict(
        retain_stories=[story for story, _ in retain_stories],
        forget_stories=[story for story, _ in forget_stories],
        num_stories=num_token_freq_calculate_stories,
        truncate_at=None,
        num_synthetic_tokens_retain=20,
        num_synthetic_tokens_forget=1,
        scale=1.0,
        bias=-4.0,
        tokenizer=enc,
        device="cpu",
    )

    freq_based_rule, info = get_token_freq_masking_rule(**token_freq_kwargs)  # type: ignore
    token_rule = (
        freq_based_rule  # Use freq_based_rule for both pure and freq_based_masking
    )

    def full_seq_mask_rule(input_ids_and_labels):
        input_ids, labels = input_ids_and_labels
        seq_length = input_ids.shape[1]
        device = input_ids.device
        """
        0 means should be in forget set
        1 means should be in retain set
        """
        return labels.unsqueeze(1).repeat(1, seq_length).to(device)

    def unmasked_retain_mask_rule(input_ids_and_labels):
        input_ids, labels = input_ids_and_labels
        original_mask = token_rule(input_ids)
        is_retain = full_seq_mask_rule((input_ids, labels))
        return torch.cat(
            [
                torch.maximum(is_retain[:, :-1], original_mask),
                torch.ones_like(original_mask[:, [0]]),
            ],
            dim=1,
        )

    mask_rule = (
        unmasked_retain_mask_rule
        if prefix != "full_seq_masking"
        else full_seq_mask_rule
    )

    def process(example):
        ids = enc.encode(example["text"])
        label = example["label"]
        ids.append(enc.eos_token_id)
        mask_ids = (
            mask_rule((torch.tensor(ids).unsqueeze(0), torch.tensor([label])))
            .squeeze(0)
            .tolist()
        )
        return {"ids": ids, "mask_ids": mask_ids, "len": len(ids)}

    # Process forget and retain datasets separately
    tokenized_forget = forget_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing forget data",
        num_proc=num_proc,
    )
    tokenized_retain = retain_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing retain data",
        num_proc=num_proc,
    )

    # Create train/validation splits for each dataset
    forget_splits = tokenized_forget.train_test_split(
        test_size=0.001, seed=11111, shuffle=True
    )
    retain_splits = tokenized_retain.train_test_split(
        test_size=0.001, seed=11111, shuffle=True
    )

    # Rename test splits to validation
    forget_splits["validation"] = forget_splits.pop("test")
    retain_splits["validation"] = retain_splits.pop("test")

    # Create the final dataset splits
    split_dataset = {
        "only_forget_train": forget_splits["train"],
        "only_forget_validation": forget_splits["validation"],
        "only_retain_train": retain_splits["train"],
        "only_retain_validation": retain_splits["validation"],
    }

    # For pure prefix, only use retain data for main splits
    if prefix == "pure":
        split_dataset["train"] = retain_splits["train"]
        split_dataset["validation"] = retain_splits["validation"]
    else:
        # For other maskings, combine forget and retain data
        split_dataset["train"] = concatenate_datasets(
            [forget_splits["train"], retain_splits["train"]]
        ).shuffle(seed=11111)
        split_dataset["validation"] = concatenate_datasets(
            [forget_splits["validation"], retain_splits["validation"]]
        ).shuffle(seed=11111)

    # Write datasets to files
    for split, dset in split_dataset.items():
        if len(dset) == 0:
            continue
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        file_name_str = f"{prefix}_tinystories_{split}.bin"
        mask_file_name_str = f"{prefix}_tinystories_{split}_masks.bin"
        filename = os.path.join(os.path.dirname(__file__), file_name_str)
        mask_filename = os.path.join(os.path.dirname(__file__), mask_file_name_str)

        for fname, seq_type, dtype in [
            (filename, "ids", np.uint16),
            (mask_filename, "mask_ids", np_type_of_mask),
        ]:
            arr = np.memmap(fname, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {fname}"):
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch[seq_type])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
