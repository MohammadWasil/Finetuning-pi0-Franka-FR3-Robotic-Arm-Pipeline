# Robotic Arm VLA Pipeline

Download Dataset: https://huggingface.co/datasets/wasil11/Franka-FR3-Unity3d-Simulation-Ball-Pick-And-Place

A simulator-agnostic end-to-end pipeline for converting robotic manipulation trajectories into **LeRobot** format, publishing datasets to **Hugging Face**, fine-tuning **π₀ (`openpi`)** Vision-Language-Action (VLA) models, and executing closed-loop inference.

## Setup (Poetry)

# 1. Clone and install everything automatically
```bash
cd robotic-arm-VLA-pipeline
bash installation.sh
```

## Usage

### Convert Raw Trajectories to LeRobot & Push to Hugging Face
Converts synchronized multi-camera frames and 8-DOF joint state/action trajectories into LeRobot v2.1 format:

```bash
poetry run python scripts/1_convert_to_lerobot.py \
  --data_dir "DATA_PATH/dataset_fr3_pick_place" \
  --repo_id "wasil11/Franka-FR3-Unity3d-Simulation-Ball-Pick-And-Place"
  --robot_type "franka_fr3"
```

### Compute normalization statistics
```bash
cd third_party/openpi
../../.venv/bin/python scripts/compute_norm_stats.py --config-name pi0_fr3_unity_low_mem_finetune
```

### Fine-tune π₀ (add WANDB_MODE=disabled if not logging into W&B)
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 ../../.venv/bin/python scripts/train.py pi0_fr3_unity_low_mem_finetune --exp-name=fr3_pick_place --overwrite
```

## Related Resources
- **Simulation Environment:** Unity 3D Franka FR3 Physics & IK Simulation [Unity3d-Franka-FR3-Robotics-Simulation-VLA](https://github.com/MohammadWasil/Unity3d-Franka-FR3-Robotics-Simulation-VLA)
- **Dataset:** The final dataset is uploaded on Hugging Face repo: https://huggingface.co/datasets/wasil11/Franka-FR3-Unity3d-Simulation-Ball-Pick-And-Place
