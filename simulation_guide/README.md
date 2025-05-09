# Using gym_hil Simulation Environments with LeRobot

This guide explains how to use the `gym_hil` simulation environments as an alternative to real robots when working with the LeRobot framework for Human-In-the-Loop (HIL) reinforcement learning.

`gym_hil` is a package that provides Gymnasium-compatible simulation environments specifically designed for Human-In-the-Loop reinforcement learning. These environments allow you to:

- Train policies in simulation to test the RL stack before training on real robots
- Collect demonstrations in sim using external devices like gamepads or keyboards
- Perform human interventions during policy learning

Currently, the main environment is a Franka Panda robot simulation based on MuJoCo, with tasks like picking up a cube.

## 1. Installation

First, install the `gym_hil` package within the LeRobot environment:

```bash
pip install gym_hil

# Or in LeRobot
cd lerobot
pip install -e .[hilserl]
```

## 2. Configuration

To use `gym_hil` with LeRobot, you need to create a configuration file. An example is provided in `gym_hil_env.json`. Key configuration sections include:

### 2.1 Environment Type and Task

```json
{
    "type": "hil",
    "name": "franka_sim",
    "task": "PandaPickCubeGamepad-v0",
    "device": "cuda"
}
```

Available tasks:
- `PandaPickCubeBase-v0`: Basic environment
- `PandaPickCubeGamepad-v0`: With gamepad control
- `PandaPickCubeKeyboard-v0`: With keyboard control

### 2.2 Gym Wrappers Configuration

```json
"wrapper": {
    "gripper_penalty": -0.02,
    "control_time_s": 15.0,
    "use_gripper": true,
    "fixed_reset_joint_positions": [0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785],
    "ee_action_space_params": {
        "x_step_size": 0.025,
        "y_step_size": 0.025,
        "z_step_size": 0.025,
        "bounds": {
            "max": [0.6, 0.3, 0.5],
            "min": [0.2, -0.3, 0.0]
        },
        "control_mode": "gamepad"
    }
}
```

Important parameters:
- `gripper_penalty`: Penalty for excessive gripper movement
- `use_gripper`: Whether to enable gripper control
- `ee_action_space_params`: Settings for end-effector movement
- `control_mode`: Set to "gamepad" to use a gamepad controller

## 3. Running with HIL RL of LeRobot

### 3.1 Basic Usage

To run the environment, set mode to null:

```python
python lerobot/scripts/server/gym_manipulator.py --config_path path/to/gym_hil_env.json
```

### 3.2 Recording a Dataset

To collect a dataset, set the mode to `record` whilst defining the repo_id and number of episodes to record:

```python
python lerobot/scripts/server/gym_manipulator.py --config_path path/to/gym_hil_env.json
```

### 3.3 Training a Policy

To train a policy, checkout the example json in `train_gym_hil_env.json` and run the actor and learner servers:

```python
python lerobot/scripts/server/actor_server.py --config_path path/to/train_gym_hil_env.json
```

In a different terminal, run the learner server:

```python 
python lerobot/scripts/server/learner_server.py --config_path path/to/train_gym_hil_env.json
```

The simulation environment provides a safe and repeatable way to develop and test your Human-In-the-Loop reinforcement learning components before deploying to real robots. 