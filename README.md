# Robotic Arm VLA Pipeline

Download Dataset: https://huggingface.co/datasets/wasil11/Franka-FR3-Unity3d-Simulation-Ball-Pick-And-Place

A simulator-agnostic end-to-end pipeline for converting robotic manipulation trajectories into **LeRobot** format, publishing datasets to **Hugging Face**, fine-tuning **π₀ (`openpi`)** Vision-Language-Action (VLA) models, and executing closed-loop inference.

## Setup (Poetry)

1. Install dependencies locally (lightweight conversion & inference bridge groups):
   ```bash
   poetry install
   ```
2. Log in to Hugging Face with a Write token / upload/Download data:
   ```bash
   poetry run huggingface-cli login
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

## Related Resources
- **Simulation Environment:** Unity 3D Franka FR3 Physics & IK Simulation [Unity3d-Franka-FR3-Robotics-Simulation-VLA](https://github.com/MohammadWasil/Unity3d-Franka-FR3-Robotics-Simulation-VLA)
- **Dataset:** The final dataset is uploaded on Hugging Face repo: https://huggingface.co/datasets/wasil11/Franka-FR3-Unity3d-Simulation-Ball-Pick-And-Place
