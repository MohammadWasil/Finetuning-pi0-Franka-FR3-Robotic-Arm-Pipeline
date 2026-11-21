# Robotic Arm VLA Pipeline

A simulator-agnostic end-to-end pipeline for converting robotic manipulation trajectories into **LeRobot** format, publishing datasets to **Hugging Face**, fine-tuning **π₀ (`openpi`)** Vision-Language-Action (VLA) models, and executing closed-loop inference.

## Setup (Poetry)

1. Install dependencies locally (lightweight conversion & inference bridge groups):
   ```bash
   poetry install
   ```
2. Log in to Hugging Face with a Write token:
   ```bash
   poetry run huggingface-cli login
   ```

## Usage

### 1. Convert Raw Trajectories to LeRobot & Push to Hugging Face
Converts synchronized multi-camera frames and 8-DOF joint state/action trajectories into LeRobot v2.1 format:

```bash
poetry run python scripts/1_convert_to_lerobot.py \
  --data_dir "DATA_PATH/dataset_fr3_pick_place" \
  --repo_id "HUGGING_FACE_ID" \
  --robot_type "franka_fr3"
```

## Related Resources
- **Simulation Environment:** Unity 3D Franka FR3 Physics & IK Simulation [Unity3d-Franka-FR3-Robotics-Simulation-VLA](https://github.com/MohammadWasil/Unity3d-Franka-FR3-Robotics-Simulation-VLA)
- **Dataset:** Franka FR3 pick and place ball in Unity3D to be uploaded soon in Hugging Face *(uploading soon)*
