import argparse
import asyncio
from datetime import datetime
import os
from typing import Dict, List
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch multiple training runs in parallel"
    )
    parser.add_argument("n", type=int, help="Number of parallel runs to launch")
    parser.add_argument("save_dir", type=str, help="Directory to save all runs")
    parser.add_argument(
        "base_id", type=str, help="Starting ID for runs (will append 1,2,3... to this)"
    )
    parser.add_argument(
        "run_type",
        choices=[
            "erac_model",
            "base_model",
            "pure_model",
            "expanded_base_model",
            "rmu_model",
        ],
        help="Type of model to run",
    )
    parser.add_argument(
        "--at-a-time", type=int, default=None, help="Maximum number of concurrent runs"
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=None, help="GPUs to limit training to"
    )
    parser.add_argument(
        "--neg-lr", type=float, help="Learning rate for negative components"
    )
    parser.add_argument(
        "--residual-coherence-nth-step",
        type=int,
        help="Steps between residual coherence calculations",
    )
    parser.add_argument(
        "--l1-coeff", type=float, default=0.0, help="L1 regularization coefficient"
    )
    parser.add_argument(
        "--compile", type=int, default=0, help="Whether to compile the model"
    )
    parser.add_argument(
        "--do-retrain-evals",
        type=bool,
        default=True,
        help="Whether to do retrain evaluations",
    )
    return parser.parse_args()


async def run_training_process(
    run_idx: int,
    base_args: Dict,
) -> int:
    """Run a single training process asynchronously"""

    # Create a unique ID for both runs_id and model_save_name
    unique_id = f"{base_args['base_id']}{run_idx + 1}"

    cmd = [
        sys.executable,  # Use current Python interpreter
        "train.py",
        unique_id,  # Use same ID for both runs_id and model_save_name
        base_args["save_dir"],
        unique_id,  # Use same ID for both runs_id and model_save_name
        base_args["run_type"],
    ]

    # Add optional arguments
    if base_args.get("gpus"):
        cmd.extend(["--gpus_to_limit_to"] + [str(gpu) for gpu in base_args["gpus"]])
    if base_args.get("neg_lr"):
        cmd.extend(["--neg-lr", str(base_args["neg_lr"])])
    if base_args.get("residual_coherence_nth_step"):
        cmd.extend(
            [
                "--residual_coherence_nth_step",
                str(base_args["residual_coherence_nth_step"]),
            ]
        )
    if base_args.get("l1_coeff"):
        cmd.extend(["--l1-coeff", str(base_args["l1_coeff"])])
    if base_args.get("compile"):
        cmd.extend(["--compile", str(base_args["compile"])])
    if base_args.get("do_retrain_evals"):
        cmd.extend(["--do-retrain-evals", str(base_args["do_retrain_evals"])])

    # Create process
    process = await asyncio.create_subprocess_exec(
        *cmd,
        # Inherit parent process stdout/stderr
        stdout=None,
        stderr=None,
    )

    print(f"Started process {run_idx + 1} with PID {process.pid}")

    # Wait for completion
    await process.wait()
    print(
        f"\nProcess {run_idx + 1} (PID {process.pid}) completed with return code {process.returncode}"
    )

    return process.returncode


async def launch_training_runs(args: argparse.Namespace) -> List[int]:
    """Launch all training runs with staggered starts and concurrency control"""
    os.makedirs(args.save_dir, exist_ok=True)

    # Create semaphore if at-a-time is specified
    sem = asyncio.Semaphore(args.at_a_time if args.at_a_time else args.n)

    base_args = {
        "save_dir": args.save_dir,
        "base_id": args.base_id,
        "run_type": args.run_type,
        "gpus": args.gpus,
        "neg_lr": args.neg_lr,
        "residual_coherence_nth_step": args.residual_coherence_nth_step,
        "l1_coeff": args.l1_coeff,
        "compile": args.compile,
        "do_retrain_evals": args.do_retrain_evals,
    }

    async def controlled_run(i: int) -> int:
        async with sem:  # Wait for semaphore before starting new process
            return await run_training_process(i, base_args)

    tasks = []
    for i in range(args.n):
        # Create task for this run
        task = asyncio.create_task(controlled_run(i))
        tasks.append(task)

        # Stagger next launch unless it's the last process
        if i < args.n - 1:
            await asyncio.sleep(100)

    # Wait for all processes to complete and collect return codes
    return await asyncio.gather(*tasks)


async def main():
    args = parse_args()
    try:
        return_codes = await launch_training_runs(args)
        print("\nAll processes completed!")
        print("Return codes:", return_codes)

        # Check for any non-zero return codes
        if any(code != 0 for code in return_codes):
            print("\nWarning: Some processes returned non-zero exit codes")
            sys.exit(1)
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
