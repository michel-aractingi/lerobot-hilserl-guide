# Training a Reward Classifier with LeRobot

This guide explains how to train a reward classifier for human-in-the-loop reinforcement learning implementation of  LeRobot. Reward classifiers learn to predict the reward value given a state which can be used in an RL setup to train a policy.


The reward classifier implementation in `modeling_classifier.py` uses a pretrained vision model to process the images. It can output either a single value for binary rewards to predict success/fail cases or multiple values for multi-class settings.

## 1. Collecting a Dataset
Before training, you need to collect a dataset with labeled examples. The `record_dataset` function in `gym_manipulator.py` enables the process of collecting a dataset of observations, actions, and rewards.

To collect a dataset, you need to modeify some parameters in the environment configuration based on HILSerlRobotEnvConfig.

```python
python lerobot/scripts/server/gym_manipulator.py --config_path lerobot/configs/reward_classifier_train_config.json
```

### Key Parameters for Data Collection:

- **mode**: set it to "record" to collect a dataset
- **repo_id**: "hf_username/dataset_name", name of the dataset and repo on the hub
- **num_episodes**: Number of episodes to record
- **number_of_steps_after_success**: Number of additional frames to record after a success (reward=1) is detected
- **fps**: Number of frames per second to record
- **push_to_hub**: Whether to push the dataset to the hub

The `number_of_steps_after_success` parameter is crucial as it allows you to collect more positive examples. When a success is detected, the system will continue recording for the specified number of steps while maintaining the reward=1 label. Otherwise, there won't be enough states in the dataset labeled to 1 to train a good classifier.

Example configuration section for data collection:

```json
{
    "mode": "record",
    "repo_id": "hf_username/dataset_name",
    "dataset_root": "data/your_dataset",
    "num_episodes": 20,
    "push_to_hub": true,
    "fps": 10,
    "number_of_steps_after_success": 15
}
```

## 2. Reward Classifier Configuration

The reward classifier is configured using `configuration_classifier.py`. Here are the key parameters:

- **model_name**: Base model architecture (e.g., we mainly use "helper2424/resnet10")
- **model_type**: "cnn" or "transformer"
- **num_cameras**: Number of camera inputs
- **num_classes**: Number of output classes (typically 2 for binary success/failure)
- **hidden_dim**: Size of hidden representation
- **dropout_rate**: Regularization parameter
- **learning_rate**: Learning rate for optimizer

Example configuration from `reward_classifier_train_config.json`:

```json
{
  "policy": {
    "type": "reward_classifier",
    "model_name": "helper2424/resnet10",
    "model_type": "cnn",
    "num_cameras": 2,
    "num_classes": 2,
    "hidden_dim": 256,
    "dropout_rate": 0.1,
    "learning_rate": 1e-4,
    "device": "cuda",
    "use_amp": true,
    "input_features": {
      "observation.images.front": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      },
      "observation.images.side": {
        "type": "VISUAL",
        "shape": [3, 128, 128]
      }
    }
  }
}
```

## 3. Training the Classifier

To train the classifier, use the `train.py` script with your configuration:

```bash
python lerobot/scripts/train.py --config_path lerobot/configs/reward_classifier_train_config.json
```

## 4. Deploying and Testing the Model

To use your trained reward classifier, configure the `HILSerlRobotEnvConfig` to use your model:

```python
env_config = HILSerlRobotEnvConfig(
    reward_classifier_pretrained_path="path_to_your_pretrained_trained_model",
    # Other environment parameters
)
```
or set the argument in the json config file.

```json
{
    "reward_classifier_pretrained_path": "path_to_your_pretrained_model"
}
```

Run gym_manipulator.py to test the model.
```bash
python lerobot/scripts/server/gym_manipulator.py --config_path lerobot/configs/env_config.json
```

The reward classifier will automatically provide rewards based on the visual input from the robot's cameras.

## Example Workflow

1. **Create the configuration files**:
   Create the necessary json configuration files for the reward classifier and the environment. Check the `json_examples` directory for examples.

2. **Collect a dataset**:
   ```bash
   python lerobot/scripts/server/gym_manipulator.py --config_path lerobot/configs/env_config.json
   ```

3. **Train the classifier**:
   ```bash
   python lerobot/scripts/train.py --config_path lerobot/configs/reward_classifier_train_config.json
   ```

4. **Test the classifier**:
   ```bash
   python lerobot/scripts/server/gym_manipulator.py --config_path lerobot/configs/env_config.json
   ```
