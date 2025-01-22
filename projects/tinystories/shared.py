from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from dataclasses_json import dataclass_json


def get_type_of_mask(masking_type: str):
    if masking_type == "freq_based_masking":
        return np.float16, torch.float16
    if masking_type == "pure":
        return np.float16, torch.float16
    if masking_type == "full_seq_masking":
        return np.float16, torch.float16
    else:
        raise ValueError("no other rules implemented yet")


@dataclass_json
@dataclass
class RunTypeConfig:
    label: str

    expand_model: bool
    use_gradient_routing: bool

    # Partial oversight (TODO not implemented yet)
    forget_data_labeling_percentage: float
    drop_labeled_forget_data: bool
    drop_unlabeled_forget_data: bool
    sort_forget_data_by_label: bool


@dataclass_json
@dataclass
class RunData:
    forget_loss_before_contract: float
    retain_loss_before_contract: float
    forget_loss_after_contract: float
    retain_loss_after_contract: float
    forget_retrain_losses: List[List[float]]
    retain_retrain_losses: List[List[float]]

    num_stories_retrain: List[
        int
    ]  # this describes the number of updates in each of the inner retain lists

    metadata: RunTypeConfig
    model_save_name: str
