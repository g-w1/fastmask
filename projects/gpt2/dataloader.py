import glob

import numpy as np
import torch


# dataloader code modified from https://github.com/karpathy/llm.c/blob/bd457aa19bdb7c0776725f05fe9ecb692558aed8/train_gpt2.py#L329
# thanks Karpathy!
def _peek_data_shard(filename):
    data = np.memmap(filename, dtype=np.uint16, mode="r")
    return len(data)


class DistributedDataLoader:
    def __init__(
        self,
        filename_pattern,
        B,
        T,
        process_rank,
        num_processes,
        mask_ids_filename=None,
        types_of_mask=None,
    ):
        """
        Assumes self.mask_ids_filename has the *exact* same structure/size as the files in self.filename_pattern.
        """
        assert bool(mask_ids_filename) == bool(
            types_of_mask
        ), "mask_ids_filename and types_of_mask must be both None or both not None"
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        self.mask_ids_files = None
        if mask_ids_filename is not None:
            self.types_of_mask = types_of_mask
            self.mask_ids_files = sorted(glob.glob(mask_ids_filename))
            assert (
                len(self.mask_ids_files) > 0
            ), f"did not find any files that match the pattern {mask_ids_filename}"
            assert len(self.mask_ids_files) == len(
                self.files
            ), "number of mask ids files must match number of data shards"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(f"DataLoader: total number of tokens: {ntok_total:,} across {self.files}")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)  # type: ignore
        self.current_position = self.process_rank * self.B * self.T

    def next_batch(self):
        B = self.B
        T = self.T
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        tokens = np.memmap(self.files[self.current_shard], dtype=np.uint16, mode="r")  # type: ignore
        buf = tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int64), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets

        if self.mask_ids_files is not None:
            mask_ids = np.memmap(
                self.mask_ids_files[self.current_shard],  # type: ignore
                dtype=self.types_of_mask[0],  # type: ignore
                mode="r",
            )
            mask_ids_buf = mask_ids[
                self.current_position : self.current_position + B * T + 1
            ]
            mask_ids_buf = torch.tensor(
                mask_ids_buf.astype(self.types_of_mask[0]),  # type: ignore
                dtype=self.types_of_mask[1],  # type: ignore
            )
            mask_ids = (mask_ids_buf[:-1]).view(B, T)
            # advance the start pointer in current shard
            self.current_position += B * T * self.num_processes
            # if loading the next batch would be out of bounds advance the shard
            if self.current_position + (B * T * self.num_processes + 1) > len(tokens):
                self.advance()
            return x, y, mask_ids
        else:
            # advance the start pointer in current shard
            self.current_position += B * T * self.num_processes
            # if loading the next batch would be out of bounds advance the shard
            if self.current_position + (B * T * self.num_processes + 1) > len(tokens):
                self.advance()
            return x, y
