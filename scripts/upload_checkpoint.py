from pathlib import Path
import shutil
import tempfile
import tyro
from huggingface_hub import HfApi


def main(
    checkpoint_root: Path = Path("checkpoints/pi0_fr3_unity_low_mem_finetune/fr3_v1"),
    repo_id: str = "wasil11/pi0-franka-fr3-pick-and-place-lora",
    step: int | None = None,
    include_train_state: bool = False,
    train_log: Path = Path("train.log"),
    private: bool = False,
) -> None:
    """Upload the fine-tuned Pi-0 checkpoint directory to Hugging Face Hub."""
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    # Auto-detect the latest numerical step folder (9999 or 10000) if not specified
    if step is None:
        step_dirs = [d for d in checkpoint_root.iterdir() if d.is_dir() and d.name.isdigit()]
        if not step_dirs:
            raise FileNotFoundError(f"No numbered checkpoint folders found in {checkpoint_root}")
        latest_dir = max(step_dirs, key=lambda d: int(d.name))
    else:
        latest_dir = checkpoint_root / str(step)
        if not latest_dir.exists():
            raise FileNotFoundError(f"Requested checkpoint step not found: {latest_dir}")

    step_name = latest_dir.name
    print(f"Preparing step '{step_name}' from {latest_dir} for upload to Hugging Face Hub")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        staged_step_dir = Path(tmp_dir) / step_name
        ignore = None if include_train_state else shutil.ignore_patterns("train_state")
        shutil.copytree(latest_dir, staged_step_dir, ignore=ignore)

        print(f"Uploading '{step_name}/' to https://huggingface.co{repo_id}")
        api.upload_folder(
            folder_path=str(staged_step_dir),
            path_in_repo=step_name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload fine-tuned Pi-0 LoRA checkpoint (step {step_name})",
        )

    if train_log.exists():
        print(f"Uploading {train_log}")
        api.upload_file(
            path_or_fileobj=str(train_log),
            path_in_repo="train.log",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload training log",
        )

    print(f"Done! Verify files at: https://huggingface.co{repo_id}/tree/main")


if __name__ == "__main__":
    tyro.cli(main)
