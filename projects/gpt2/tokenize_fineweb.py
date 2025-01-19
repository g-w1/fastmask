# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
import transformers

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

# enc = tiktoken.get_encoding("gpt2")
enc = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

sprinkle_wmdp_in_nx = 0.5

import json
from typing import Dict, List, Literal

import torch as t
from datasets import load_dataset


# https://github.com/centerforaisafety/wmdp/blob/main/rmu/utils.py#L70 for min_len
def load_wmdp_data(
    min_len=50,
    use_wikitext_retain=True,
    section: Literal["bio", "cyber"] = "cyber",
    shuffle=True,
) -> Dict[str, List[str]]:
    retain_filename = f"{section}-retain-corpus.jsonl"
    remove_filename = f"{section}-forget-corpus.jsonl"
    wmdp_dataset_path = "../../.wmdp/wmdp-corpora/"
    retain_corpus_path = wmdp_dataset_path + retain_filename
    remove_corpus_path = wmdp_dataset_path + remove_filename
    retain_corpus = []
    remove_corpus = []
    if use_wikitext_retain:
        raw_data = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="train[:400000]"
        )
        for x in raw_data:
            if len(x["text"]) > min_len:
                retain_corpus.append(str(x["text"]))
    else:
        for line in open(retain_corpus_path):
            assert type(line) is str
            if len(line) > min_len:
                retain_corpus.append(json.loads(line))
    for line in open(remove_corpus_path):
        if "bio-forget-corpus" in remove_corpus_path:
            raw_text = json.loads(line)["text"]
        else:
            raw_text = line
        if len(raw_text) > min_len:
            remove_corpus.append(str(raw_text))
    if shuffle:
        # deterministic shuffle
        import random

        random.seed(42)
        random.shuffle(retain_corpus)
        random.shuffle(remove_corpus)
    return {"retain": retain_corpus, "forget": remove_corpus}


if __name__ == "__main__":
    if sprinkle_wmdp_in_nx > 0:
        print(
            f"SPRINKLING WMDP BIO FORGET SET IN {sprinkle_wmdp_in_nx}x TIMES to train set"
        )
        print("loading wmdp data...")
        wmdp_forget_data = load_wmdp_data(section="bio", min_len=1000)[
            "forget"
        ]  # a list of strings

        # Convert wmdp_forget_data into a dataset

        if sprinkle_wmdp_in_nx > 1:
            wmdp_dataset = Dataset.from_dict({"text": wmdp_forget_data})
            # Repeat the dataset sprinkle_wmdp_in_nx times
            repeated_wmdp_datasets = [wmdp_dataset for _ in range(sprinkle_wmdp_in_nx)]
            wmdp_dataset = concatenate_datasets(repeated_wmdp_datasets)
        else:
            wmdp_dataset = Dataset.from_dict(
                {
                    "text": wmdp_forget_data[
                        : int(len(wmdp_forget_data) * sprinkle_wmdp_in_nx)
                    ]
                }
            )
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    num_tokens = "10BT"
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=f"sample-{num_tokens}",
        num_proc=num_proc_load_dataset,
    )

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

    new_split = split_dataset["train"].train_test_split(
        test_size=0.5, seed=2357, shuffle=True
    )
    new_split["residual_coherence_set"] = new_split.pop("test")

    split_dataset["train"] = new_split["train"]
    split_dataset["residual_coherence_set"] = new_split["residual_coherence_set"]

    # concat the 'train' split with the wmdp dataset
    if sprinkle_wmdp_in_nx > 0:
        split_dataset["train"] = concatenate_datasets(
            [split_dataset["train"], wmdp_dataset]
        ).shuffle(seed=42)  # type: ignore

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function
    def process(example):
        ids = enc.encode(example["text"])  # encode_ordinary ignores any special tokens
        ids.append(
            enc.eos_token_id
        )  # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        file_name_str = (
            f"{split}.bin"
            if sprinkle_wmdp_in_nx == 0
            else f"fineweb_{num_tokens}_{split}_wmdp_{sprinkle_wmdp_in_nx}.bin"
        )
        filename = os.path.join(os.path.dirname(__file__), file_name_str)
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
