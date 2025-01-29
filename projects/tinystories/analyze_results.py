# %%
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import dataclasses
from typing import List, Dict, Tuple


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
    min_forget_retrain_loss: float
    min_forget_retrain_loss_ci: float


def ci_95(data):
    std = np.std(data)
    return 1.96 * std / np.sqrt(len(data))


def stats_from_dir(dir):
    forget_losses_before_contract = []
    forget_losses_after_contract = []
    retain_losses_before_contract = []
    retain_losses_after_contract = []
    forget_losses_after_coherence = []
    retain_losses_after_coherence = []
    min_forget_retrain_losses = []

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
                min_forget_retrain_losses.append(min(forget_retraining_series))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            raise e

    return Stats(
        forget_loss_before_contract=np.mean(forget_losses_before_contract),
        forget_loss_before_contract_ci=ci_95(forget_losses_before_contract),
        forget_loss_after_contract=np.mean(forget_losses_after_contract),
        forget_loss_after_contract_ci=ci_95(forget_losses_after_contract),
        retain_loss_before_contract=np.mean(retain_losses_before_contract),
        retain_loss_before_contract_ci=ci_95(retain_losses_before_contract),
        retain_loss_after_contract=np.mean(retain_losses_after_contract),
        retain_loss_after_contract_ci=ci_95(retain_losses_after_contract),
        min_forget_retrain_loss=np.mean(min_forget_retrain_losses),
        min_forget_retrain_loss_ci=ci_95(min_forget_retrain_losses),
        forget_loss_after_coherence=np.mean(forget_losses_after_coherence),
        forget_loss_after_coherence_ci=ci_95(forget_losses_after_coherence),
        retain_loss_after_coherence=np.mean(retain_losses_after_coherence),
        retain_loss_after_coherence_ci=ci_95(retain_losses_after_coherence),
    )


class BarPlotConfig:
    def __init__(self, width: float = 0.15):
        self.width = width
        self._bars = []
        # Using a nicer color palette (Tableau-style colors)
        self.colors = plt.cm.Set2(np.linspace(0, 1, 8))
        # Define canonical stage-to-color mapping
        self._stage_colors = {
            "Before Contract": self.colors[0],
            "After Contract": self.colors[1],
            "After Coherence": self.colors[2],
            "Min Forget Retrain": self.colors[3],
        }

    def add_bar(
        self, values: List[float], errors: List[float], label: str, offset: float = 0
    ):
        self._bars.append((values, errors, label, offset))
        return self

    def plot(self, ax: plt.Axes, x: np.ndarray, ylim: Tuple[float, float] = (0, 2.0)):
        for values, errors, label, offset in self._bars:
            ax.bar(
                x + offset * self.width,
                values,
                self.width,
                label=label,
                yerr=errors,
                color=self._stage_colors[
                    label
                ],  # Use the canonical color for this stage
                capsize=3,
            )
        ax.set_ylim(*ylim)


def create_loss_plot(
    results: Dict[str, Stats],
    dirs: List[str],
    metric_group: List[str],
    nice_dir_names: Dict[str, str],
    figsize: Tuple[int, int] = (8, 6),
    subtract_base=False,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)

    labels = [nice_dir_names.get(dir, dir) for dir in dirs]
    metric_name = metric_group[0].split("_")[0]
    x = np.arange(len(labels))

    before_contract = [results[dir].__dict__[metric_group[0]] for dir in dirs]
    before_contract_ci = [
        results[dir].__dict__[metric_group[0] + "_ci"] for dir in dirs
    ]
    after_contract = [results[dir].__dict__[metric_group[1]] for dir in dirs]
    after_contract_ci = [results[dir].__dict__[metric_group[1] + "_ci"] for dir in dirs]
    after_coherence = [
        results[dir].__dict__[f"{metric_name}_loss_after_coherence"] for dir in dirs
    ]
    after_coherence_ci = [
        results[dir].__dict__[f"{metric_name}_loss_after_coherence_ci"] for dir in dirs
    ]
    min_forget_retrain = [results[dir].min_forget_retrain_loss for dir in dirs]
    min_forget_retrain_ci = [results[dir].min_forget_retrain_loss_ci for dir in dirs]

    if subtract_base:
        before_contract = [x - before_contract[0] for x in before_contract]
        after_contract = [x - after_contract[0] for x in after_contract]
        after_coherence = [x - after_coherence[0] for x in after_coherence]
        min_forget_retrain = [x - min_forget_retrain[0] for x in min_forget_retrain]

    plot_config = BarPlotConfig()

    if metric_name == "retain":
        plot_config.add_bar(before_contract, before_contract_ci, "Before Contract", -1)
        plot_config.add_bar(after_contract, after_contract_ci, "After Contract", 0)
        plot_config.add_bar(after_coherence, after_coherence_ci, "After Coherence", 1)
    else:
        plot_config.add_bar(before_contract, before_contract_ci, "Before Contract", -1)
        plot_config.add_bar(after_coherence, after_coherence_ci, "After Coherence", 0)
        plot_config.add_bar(
            min_forget_retrain, min_forget_retrain_ci, "Min Forget Retrain", 1
        )

    ylims = (-0.2, 0.4) if subtract_base else (0, 2)
    plot_config.plot(ax, x, ylim=ylims)

    yticks = np.arange(ylims[0], ylims[1] + 0.1, 0.1)
    ax.set_yticks(yticks)

    ax.set_ylabel("Loss")
    ax.set_title(
        f"{metric_name.capitalize()} Loss{' Deltas To Base' if subtract_base else ''}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()

    return fig, ax


# Main execution
dirs = [
    "baselines",
    "param_level_masking",
    "param_level_residual_coherence",
    "basic_50_residual_coherence_neg_1",
    "basic_25_residual_coherence_neg_0.5",
]

nice_dir_names = {
    "baselines": "Base",
    "residual_coherence_benchmarks": "Res. Coh. (act)",
    "param_level_residual_coherence": "Res. Coh. (param)",
    "activation_level_masking": "Act. Routing",
    "param_level_masking": "Param Routing",
    "basic_50_residual_coherence_neg_1": "2% res coh, -1 lr",
}

# Calculate results
results = {dir: stats_from_dir(dir) for dir in dirs}

# Style configuration

# Create plots for each metric group
for metric_group in Stats.metric_groups:
    create_loss_plot(results, dirs, metric_group, nice_dir_names, subtract_base=True)

plt.show()
