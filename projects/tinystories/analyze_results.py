# %%
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import dataclasses

dirs = [
    "baselines",
    "basic_1_residual_coherence",
    "residual_coherence_benchmarks",
    "activation_level_masking",
    "param_level_masking",
    "param_level_residual_coherence",
]


def ci_95(data):
    std = np.std(data)
    return 1.96 * std / np.sqrt(len(data))


@dataclasses.dataclass
class Stats:
    metric_groups = [
        ["forget_loss_before_contract", "forget_loss_after_contract"],
        ["retain_loss_before_contract", "retain_loss_after_contract"],
    ]
    forget_loss_before_contract: float
    forget_loss_before_contract_ci: float
    forget_loss_after_contract: float
    forget_loss_after_contract_ci: float
    retain_loss_before_contract: float
    retain_loss_before_contract_ci: float
    retain_loss_after_contract: float
    retain_loss_after_contract_ci: float
    forget_loss_after_coherence: float
    forget_loss_after_coherence_ci: float
    retain_loss_after_coherence: float
    retain_loss_after_coherence_ci: float
    min_forget_retain_loss: float
    min_forget_retain_loss_ci: float


def stats_from_dir(dir):
    forget_losses_before_contract = []
    forget_losses_after_contract = []
    retain_losses_before_contract = []
    retain_losses_after_contract = []
    forget_losses_after_coherence = []
    retain_losses_after_coherence = []
    min_forget_retain_losses = []
    for filename in os.listdir(dir):
        try:
            with open(os.path.join(dir, filename), "r") as f:
                data = json.load(f)
                forget_retraining_series = data["forget_retrain_losses"][0]
                retain_retraining_series = data["retain_retrain_losses"][0]
                forget_losses_before_contract.append(
                    data["forget_loss_before_contract"]
                )
                forget_losses_after_contract.append(data["forget_loss_after_contract"])
                retain_losses_before_contract.append(
                    data["retain_loss_before_contract"]
                )
                retain_losses_after_contract.append(data["retain_loss_after_contract"])
                forget_losses_after_coherence.append(forget_retraining_series[0])
                retain_losses_after_coherence.append(retain_retraining_series[0])
                min_forget_retain_losses.append(min(forget_retraining_series))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            raise e

    datas = Stats(
        forget_loss_before_contract=np.mean(forget_losses_before_contract),
        forget_loss_before_contract_ci=ci_95(forget_losses_before_contract),
        forget_loss_after_contract=np.mean(forget_losses_after_contract),
        forget_loss_after_contract_ci=ci_95(forget_losses_after_contract),
        retain_loss_before_contract=np.mean(retain_losses_before_contract),
        retain_loss_before_contract_ci=ci_95(retain_losses_before_contract),
        retain_loss_after_contract=np.mean(retain_losses_after_contract),
        retain_loss_after_contract_ci=ci_95(retain_losses_after_contract),
        min_forget_retain_loss=np.mean(min_forget_retain_losses),
        min_forget_retain_loss_ci=ci_95(min_forget_retain_losses),
        forget_loss_after_coherence=np.mean(forget_losses_after_coherence),
        forget_loss_after_coherence_ci=ci_95(forget_losses_after_coherence),
        retain_loss_after_coherence=np.mean(retain_losses_after_coherence),
        retain_loss_after_coherence_ci=ci_95(retain_losses_after_coherence),
    )

    return datas


res = {}
for dir in dirs:
    res[dir] = stats_from_dir(dir)
nice_dir_names = {
    "baselines": "Base",
    "residual_coherence_benchmarks": "Res. Coh. (act)",
    "param_level_residual_coherence": "Res. Coh. (param)",
    "activation_level_masking": "Act. Routing",
    "param_level_masking": "Param Routing",
}
# make a bar chart for each of the three metrics
for metric_group in Stats.metric_groups:
    metric_name = metric_group[0].split("_")[0]
    # wide chart
    fig, ax = plt.subplots(figsize=(15, 6))
    labels = [nice_dir_names.get(dir, dir) for dir in dirs]
    before_contract = [res[dir].__dict__[metric_group[0]] for dir in dirs]
    before_contract_ci = [res[dir].__dict__[metric_group[0] + "_ci"] for dir in dirs]
    after_contract = [res[dir].__dict__[metric_group[1]] for dir in dirs]
    after_contract_ci = [res[dir].__dict__[metric_group[1] + "_ci"] for dir in dirs]
    after_coherence = [
        res[dir].__dict__[metric_name + "_loss_after_coherence"] for dir in dirs
    ]
    after_coherence_ci = [
        res[dir].__dict__[metric_name + "_loss_after_coherence_ci"] for dir in dirs
    ]
    min_forget_retain = [res[dir].min_forget_retain_loss for dir in dirs]
    min_forget_retain_ci = [res[dir].min_forget_retain_loss_ci for dir in dirs]

    x = np.arange(len(labels))
    width = 0.20

    ax.bar(
        x - 1.5 * width,
        before_contract,
        width,
        label="Before Contract",
        yerr=before_contract_ci,
    )
    ax.bar(
        x - width / 2,
        after_contract,
        width,
        label="After Contract",
        yerr=after_contract_ci,
    )
    ax.bar(
        x + width / 2,
        after_coherence,
        width,
        label="After Coherence",
        yerr=after_coherence_ci,
    )
    ax.bar(
        x + 1.5 * width,
        min_forget_retain,
        width,
        label="Min Forget Retain",
        yerr=min_forget_retain_ci,
    )

    ax.set_ylabel("Loss")
    ax.set_title(f"{metric_name} Loss Before and After Contract")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right")


# %%
