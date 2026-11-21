import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def main(
    data_dir: str = "D:/ML/RobotLearning/dataset_fr3_pick_place",
    repo_id: str = "MY_HF_USERNAME/fr3-unity-ball-pick-place",
    push_to_hub: bool = True,
    private: bool = False,
):
    raw_dir = Path(data_dir)
    with open(raw_dir / "dataset_info.json", "r") as f:
        meta = json.load(f)

    fps = int(meta.get("fps", 15))
    img_h, img_w = meta.get("image_size",)
    default_prompt = meta.get(
        "prompt", "pick up the red ball and place it in the bowl"
    )

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="franka_fr3",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    episode_dirs = sorted(raw_dir.glob("episode_*"))
    print(f"Found {len(episode_dirs)} episodes in {raw_dir}")

    state_cols = [f"state_j{i}" for i in range(7)] + ["state_gripper"]
    action_cols = [f"action_j{i}" for i in range(7)] + ["action_gripper"]

    for ep_dir in tqdm(episode_dirs, desc="Converting episodes"):
        csv_path = ep_dir / "trajectory.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if len(df) < 5:
            continue

        states = df[state_cols].to_numpy(dtype=np.float32)
        actions = df[action_cols].to_numpy(dtype=np.float32)

        for step_idx in range(len(df)):
            frame_name = f"frame_{step_idx:05d}.png"
            ext_img = np.asarray(
                Image.open(ep_dir / "exterior_image" / frame_name).convert("RGB")
            )
            wrist_img = np.asarray(
                Image.open(ep_dir / "wrist_image" / frame_name).convert("RGB")
            )

            dataset.add_frame(
                {
                    "image": ext_img,
                    "wrist_image": wrist_img,
                    "state": states[step_idx],
                    "actions": actions[step_idx],
                    "task": default_prompt,
                }
            )

        dataset.save_episode()

    if push_to_hub:
        print(f"Uploading to Hugging Face Hub: {repo_id} ...")
        dataset.push_to_hub(
            tags=["franka", "fr3", "unity", "pick-and-place", "openpi"],
            private=private,
            push_videos=False,
            license="apache-2.0",
        )
        print(f"Uploaded! View your dataset at: https://huggingface.co{repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
