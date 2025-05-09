# LeRobot Server Directory Guide

This guide provides an overview of the core components in the LeRobot server directory, including the environment wrappers, actor-learner architecture, and their interactions.

## Table of Contents

- [Real Robot Environment (gym_manipulator.py)](#environment-gym_manipulatorpy)
  - [Main Environment Class](#main-environment-class)
  - [Environment Wrappers](#environment-wrappers)
- [Environment Configuration (configs.py)](#environment-configuration-configspy)
  - [HILSerlRobotEnvConfig](#hilserlrobotenvconfigm)
  - [Action Space Configuration](#action-space-configuration)
  - [Wrapper Configuration](#wrapper-configuration)
- [Reward Classifier](#reward-classifier)
  - [Purpose and Functionality](#purpose-and-functionality)
  - [Architecture](#architecture)
  - [Training Process](#training-process-1)
  - [Integration with Environment](#integration-with-environment)
- [Robot Kinematics (kinematics.py)](#robot-kinematics-kinematicspy)
  - [Forward and Inverse Kinematics](#forward-and-inverse-kinematics)
  - [Integration with Gym Environment](#integration-with-gym-environment)
- [Control Utils (end_effector_control_utils.py)](#control-utils-end_effector_control_utilspy)
  - [Input Controllers](#input-controllers)
  - [End-Effector Control](#end-effector-control)
- [Actor (actor_server.py)](#actor-actor_serverpy)
  - [Core Functionality](#actor-core-functionality)
  - [Communication](#actor-communication)
- [Learner (learner_server.py)](#learner-learner_serverpy)
  - [Core Functionality](#learner-core-functionality)
  - [Training Process](#training-process)
- [gRPC Communication (learner_service.py)](#grpc-communication-learner_servicepy)
  - [Service Implementation](#service-implementation)
  - [Data Streaming](#data-streaming)
- [Actor-Learner Communication Flow](#actor-learner-communication-flow)

## Real Robot Environment (gym_manipulator.py)

### Main Environment Class

`HILSerlRobotEnv` is the primary environment class that implements the OpenAI Gym interface for robot control. It handles:

- Robot class initialization
- Gym-style observation and action space setup
- Step/reset functionality for environment interaction and low level control
- Observation space elements retrieval

### Environment Wrappers

The file contains several wrappers that modify the environment's behavior:

| Wrapper | Purpose |
|---------|---------|
| `AddJointVelocityToObservation` | Adds joint velocity information to the observation space |
| `AddCurrentToObservation` | Adds joint current readings to the observation space |
| `RewardWrapper` | Applies a reward classifier to provide the rewards |
| `TimeLimitWrapper` | Limits episode length to a specified time |
| `ImageCropResizeWrapper` | Crops and resizes camera images to a region of interest |
| `ConvertToLeRobotObservation` | Converts observations to the format expected by LeRobot policies |
| `KeyboardInterfaceWrapper` | Allows keyboard control during environment interaction |
| `ResetWrapper` | Customizes the environment reset behavior |
| `BatchCompitableWrapper` | Makes observations compatible with batch processing |
| **Wrappers related to end-effector control:** |
| `EEActionWrapper` | Transforms actions to end-effector space |
| `EEObservationWrapper` | Adds end-effector pose to observations |
| `GripperActionWrapper` | Converts continuous gripper actions to discrete |
| `GripperoPenaltyWrapper` | Adds penalties for unnecessary gripper actions |
| `GamepadControlWrapper` | Enables gamepad control for the robot |

The `make_robot_env()` function creates an environment with the appropriate wrappers based on configuration.
The use of these wrappers is governed by the `EnvWrapperConfig` class in the configuration file.

## Environment Configuration (configs.py)

The environment setup is driven by configuration classes that define how the robot, wrappers, and action spaces function.

### HILSerlRobotEnvConfig

The `HILSerlRobotEnvConfig` class is the main configuration container for the `HILSerlRobotEnv` environment. Key parameters include:

| Parameter | Purpose |
|-----------|---------|
| `robot` | RobotConfig instance that specifies robot hardware details |
| `wrapper` | EnvWrapperConfig instance for customizing env wrappers |
| `fps` | Control frequency in Hz (default: 10) |
| `name` | Environment name identifier |
| `mode` | Operating mode ("record", "replay", or None (during training)) |
| `task` | Task identifier string |
| `device` | Compute device for tensor operations (default: "cuda") |
| `reward_classifier_pretrained_path` | Path to pretrained reward model |

### Action Space Configuration

The `EEActionSpaceConfig` class configures the end-effector action space:

```python
@dataclass
class EEActionSpaceConfig:
    x_step_size: float        # Step size for X-axis movement
    y_step_size: float        # Step size for Y-axis movement 
    z_step_size: float        # Step size for Z-axis movement
    bounds: Dict[str, Any]    # Position limits (min/max)
    control_mode: str = "gamepad" # Enable gamepad control
```

This configuration:
- Defines movement increments for end-effector control
- Sets workspace boundaries for safety
- Enables optional gamepad control

### Wrapper Configuration

The `EnvWrapperConfig` class provides detailed control over environment wrappers:

| Parameter | Purpose |
|-----------|---------|
| `display_cameras` | Enable camera display during operation |
| `add_joint_velocity_to_observation` | Include joint velocities in observation |
| `add_current_to_observation` | Include joint current readings in observation |
| `add_ee_pose_to_observation` | Include end-effector pose in observation |
| `crop_params_dict` | Image cropping parameters for cameras |
| `resize_size` | Target size for resizing images |
| `control_time_s` | Maximum episode duration |
| `fixed_reset_joint_positions` | Joint positions for resetting the robot |
| `reset_time_s` | Time allocated for reset movements |
| `joint_masking_action_space` | Boolean mask for disabling specific joints |
| `ee_action_space_params` | End-effector action space configuration |
| `use_gripper` | Enable gripper control |
| `gripper_quantization_threshold` | Threshold for discretizing gripper actions |
| `gripper_penalty` | Penalty value for unnecessary gripper actions |
| `open_gripper_on_reset` | Open gripper during environment reset |

These configuration options allow precise customization of:
- Observation space content and formatting
- Action space behavior and constraints
- Episode timing and reset behavior
- Gripper control and penalties

The configuration system uses Python dataclasses with inheritance and defaults, making it easy to adjust individual parameters while keeping sensible defaults for the rest.

## Reward Classifier

### Purpose and Functionality

- The reward classifier is an important component in HILSerl to define the reward signal for RL training:
- **Binary Signal**: Outputs a single 0-1 signal indicating failure or success


### Architecture

The reward classifier is implemented in `modeling_classifier.py` with a straightforward architecture:

- **Pretrained Vision Encoder**: Uses existing pretrained vision models on ImageNet (ResNet10, etc.)
  - Typically frozen during training to avoid overfitting
  
- **MLP Head**: Adapts to the specific task requirements
  - Processes features from the vision encoder
  - Maps to a binary classification output
  - Simple enough to train with limited demonstration data

The architecture is configurable through the `ClassifierConfig` class in `configuration_classifier.py`, allowing parameters such as vision encoder choice, hidden layer sizes, and whether to freeze the encoder.

### Training Process

Unlike other specialized training procedures, the reward classifier uses the standard training pipeline:

1. **Dataset Collection**: Can be done either with `gym_manipulator.py` or with `control_robot.py`
   - Uses datasets different to the offline demonstration datasets that the policy is trained on
   - Success states from demonstrations are labeled as positive examples

2. **Configuration**: Check the example configuration in `reward_classifier_train_config.json`

3. **Training Commands**:
   ```bash
   python lerobot/scripts/train.py --config_path lerobot/configs/reward_classifier_train_config.json
   ```

4. **Loading the Classifier**:
   - Set the `reward_classifier_pretrained_path` in the `HILSerlRobotEnvConfig` to the path of the trained classifier

### Integration with Environment

The reward classifier integrates with the environment through the `RewardWrapper` in `gym_manipulator.py`:

```python
# Example usage in make_robot_env function
if cfg.reward_classifier_pretrained_path is not None:
    reward_classifier = Classifier.from_pretrained(cfg.reward_classifier_pretrained_path)
    env = RewardWrapper(env, reward_classifier, device=device)
```

The wrapper:
1. Intercepts environment observations after each step
2. Passes observations through the reward classifier
3. Provides the binary classification result (0 or 1) as the reward signal
4. Maintains the environment's sparse reward structure while providing useful feedback

## Robot Kinematics (kinematics.py)

### Forward and Inverse Kinematics

The `RobotKinematics` class provides mathematical tools for robot motion:

- **Forward Kinematics (FK)**: Calculates end-effector pose from joint angles
  - Supports different robot types (so100, koch, moss)
  - Handles transformations for each robot link (shoulder, humerus, forearm, wrist, gripper)
  - Computes poses for different points on the robot

- **Inverse Kinematics (IK)**: Calculates joint angles to achieve a desired end-effector pose
  - Uses gradient descent optimization
  - Supports position-only or full pose (position and orientation) targets
  - Computes Jacobians for efficient motion planning

Key mathematical operations include:
- Screw theory for rigid body transformations
- Rodrigues' rotation formula
- SE(3) error computations for pose differences

### Integration with Gym Environment

The kinematics module integrates with the gym environment through:

- The `EEActionWrapper` in gym_manipulator.py, which uses kinematics to:
  - Convert end-effector delta commands to joint space actions
  - Enforce workspace bounds for safe operation
  - Handle different control modes (absolute vs. delta)

- The `EEObservationWrapper`, which:
  - Adds end-effector pose to observations
  - Normalizes poses based on workspace limits

## Control Utils (end_effector_control_utils.py)

### Input Controllers

The file provides several controller implementations for human input:

| Controller | Description |
|------------|-------------|
| `InputController` | Base class with common interface for all controllers |
| `KeyboardController` | Maps keyboard keys to X/Y/Z axis movements |
| `GamepadController` | Maps gamepad joysticks to end-effector movements using PyGame |
| `GamepadControllerHID` | Alternative gamepad implementation using direct HID access (for MacOS)|

All controllers provide:
- Movement delta generation (dx, dy, dz)
- Intervention signaling
- Gripper commands (open/close)
- Episode status management (success/failure)

### End-Effector Control

The module enables different control modes for robot manipulation:

- **Delta End-Effector Control**: Incremental movements in Cartesian space
  - Maps controller inputs to small position changes
  - Uses inverse kinematics to convert to joint commands
  - Enforces workspace bounds for safety

- **Gym Environment Integration**:
  - `teleoperate_gym_env()`: Controls robots through the gym interface
  - Enables human demonstration collection for imitation learning
  - Supports intervention for human-in-the-loop learning

The `GamepadControlWrapper` in gym_manipulator.py uses these utilities to:
- Process gamepad inputs at environment step time
- Generate appropriate actions based on joystick positions
- Signal interventions for human-in-the-loop reinforcement learning

## Actor (actor_server.py)

### Actor Core Functionality

The actor component performs the following:

- Executes the policy in the environment to collect experiences
- Manages interaction with the robot or simulation
- Pushes collected transitions to the learner
- Receives updated policy parameters from the learner

The main execution flow is in `act_with_policy()`, which:
1. Initializes the environment and policy
2. Executes the policy to generate actions
3. Collects transitions (state, action, reward, next_state)
4. Sends transitions to the learner
5. Updates policy parameters when received from learner

### Actor Communication

Communication happens over gRPC with these main processes:

- `receive_policy`: Receives updated policy parameters from the learner
- `send_transitions`: Sends environment transitions to the learner
- `send_interactions`: Sends interaction statistics to the learner

## Learner (learner_server.py)

### Learner Core Functionality

The learner is responsible for:

- Training the policy using transitions from the actor
- Maintaining replay buffers (online and offline)
- Sending updated policy parameters to actors
- Saving checkpoints and training state

### Training Process

The main training loop in `add_actor_information_and_train()`:

1. Receives transitions from actors and adds them to the replay buffer
2. Samples batches from the replay buffer
3. Updates critic networks multiple times (UTD ratio)
4. Periodically updates the actor network and temperature parameter
5. Pushes updated policy parameters to actors
6. Logs metrics and saves checkpoints

The training employs these key mechanisms:

- **UTD Ratio**: Performs multiple critic updates per environment step
- **Offline Buffer**: Optionally maintains offline transitions for offline learning
- **Policy Update Frequency**: Controls how often the actor is updated

## gRPC Communication (learner_service.py)

### Service Implementation

The `LearnerService` class implements the gRPC service that facilitates actor-learner communication:

- Inherits from `hilserl_pb2_grpc.LearnerServiceServicer`
- Manages queues for parameters, transitions, and interaction messages
- Handles shutdown events for graceful termination
- Configures message size limits for large tensor transfers

Key configuration parameters:
- `MAX_MESSAGE_SIZE`: 4MB default for tensor transfers
- `MAX_WORKERS`: Number of concurrent gRPC server threads
- `SHUTDOWN_TIMEOUT`: Grace period for server shutdown

### Data Streaming

The service provides three main streaming endpoints:

1. **`StreamParameters`**: 
   - Streams policy parameters from learner to actors
   - Uses queue-based communication for thread/process safety
   - Implements push frequency control to avoid overloading actors

2. **`SendTransitions`**:
   - Receives experience transitions from actors
   - Handles chunked data transfer for large batches
   - Pushes received data to a queue for processing by the trainer

3. **`SendInteractions`**:
   - Receives episode statistics and metrics from actors
   - Enables monitoring of agent performance
   - Facilitates logging and visualization

The service also provides a `Ready` method for connectivity testing and handshaking between actors and learner.

## Actor-Learner Communication Flow

The system uses a distributed architecture with these data flows:

1. **Actors → Learner**: 
   - Environment transitions (state, action, reward, next_state)
   - Interaction statistics (rewards, episode information)

2. **Learner → Actors**:
   - Updated policy parameters

The communication is implemented through gRPC with persistent connections, allowing efficient transfer of potentially large tensors. 