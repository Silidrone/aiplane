# Introduction

AiPlane is an AI agent project for autonomous aircraft landing in X-Plane 12 flight simulator. This project is still in active development. The agent uses deep reinforcement learning to learn aircraft control through reinforcement learning training.

**Note**: This is a work-in-progress project. Contributions are welcome!

## Technologies Used

- **Reinforcement Learning**: SARSA (State-Action-Reward-State-Action) algorithm with ε-greedy exploration
- **Neural Networks**: PyTorch-based deep Q-network with 3 fully connected layers (12-128-128-1 architecture)
- **Flight Simulation**: X-Plane 12 with XPPython3 plugin for real-time data exchange
- **State Space**: 12-dimensional normalized feature vector including distance, altitude, deviations, and flight parameters
- **Action Space**: 784 discrete actions across 4 control surfaces (elevator, throttle, aileron, flaps)

# Installation Guide
1. Please install the xppython3 plugin before running anything: https://xppython3.readthedocs.io/en/latest/usage/installation_plugin.html
2. Open the game so the xppython3 gets initialized and generates the PythonPlugins directory.
3. In your powershell, run the following to install torch cuda (again, assuming your path is Steam installed XPLANE12): `& "C:\Program Files (x86)\Steam\steamapps\common\X-Plane 12\Resources\plugins\XPPython3\win_x64\python.exe" -m pip install torch==2.7.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118`
4. To have the files sync up with the xppython3's PythonPlugins directory (thats where they are executed), please edit settings.json such that it contains the correct absolute path to your PythonPlugins directory (it should work by default if you're on Windows and have installed XPLANE12 via Steam, and if you've properly installed the xppython3 plugin).
Please use Python 3.11.7 if you want to locally install the packages so that you have an easier time working with the IDE (this is optional).

#### Note: I am running this on Windows as I had troubles running the xppython3 plugin on Ubuntu. I will update the readme once I am able to port it to Ubuntu as well.

## Development History

This project represents the culmination of a multi-stage development process:

1. **Foundation**: Started with a complete C++ implementation of reinforcement learning framework and custom neural networks, validated through Sutton & Barto exercises and a 2D tag game.

2. **Python Migration**: Transitioned to [ai-taggame](https://github.com/Silidrone/ai-taggame) - ported the entire RL framework and tag game to Python, replacing custom neural network implementation with PyTorch for improved performance and ability to train on GPUs (CUDA).

3. **Real-World Application**: AiPlane applies the proven RL framework + PyTorch to aviation control in a realistic flight simulator environment.

# Technical Details
The state space captures essential aircraft flight parameters and environmental conditions for autonomous landing:

#### Features (12):
1. **Distance to Runway Threshold** (meters): Computed using haversine formula for great-circle distance between current GPS coordinates and runway 34R threshold (38.278404°N, 27.161163°E)
2. **Mean Sea Level Altitude** (meters): Current aircraft elevation above ground level, critical for approach path management
3. **Lateral Deviation** (meters): Perpendicular distance from runway centerline, calculated using vector projection from aircraft position to extended runway centerline
4. **Vertical Deviation** (meters): Deviation from standard 3.0° ILS glide slope, computed using trigonometric relationships and runway elevation (114.028m)
5. **Vertical Speed** (ft/min): Rate of altitude change, critical for approach stability
6. **Pitch Attitude** (degrees): Aircraft nose-up/nose-down angle from horizon
7. **Bank Angle** (degrees): Aircraft roll angle, essential for lateral control
8. **True Airspeed** (knots): Current aircraft velocity through air mass, essential for stall/overspeed detection
- +4 more (action space)

#### State Normalization:
All features undergo min-max normalization:
- **Distance**: 6000m → -1.0
- **MSL altitude**: 116m → +1.0, 450m → -1.0
- **Lateral deviation**: ±500m → ±1.0
- **Vertical deviation**: ±120m → ±1.0
- **Vertical speed**: ±10 ft/min → ±1.0
- **Pitch**: ±35° → ±1.0
- **Bank**: ±40° → ±1.0
- **Airspeed**: 25-120 knots → 0.0-1.0

### ACTION SPACE

The action space converts continuous flight controls into discrete actions to reduce computational complexity.

#### Primary Flight Controls:
1. **Elevator Control**: [-0.3 to +0.3] with 0.1 precision → 7 discrete values (pitch control)
2. **Throttle Control**: [0.0, 0.33, 0.66, 1.0] → 4 discrete values (power management)
3. **Aileron Control**: [-0.3 to +0.3] with 0.1 precision → 7 discrete values (roll control)
4. **Flap Configuration**: [0.0, 0.33, 0.66, 1.0] → 4 discrete values (progressive flap settings)

#### ANN Architecture & Hyperparameters
- **Total Action Space: 7 × 4 × 7 × 4 = 784 discrete actions**
- **Architecture Type**: Feed-forward Neural Network (DQN) with state-action value approximation
- **Input Layer**: 12 neurons (8 state features + 4 action features)
- **Hidden Layer 1**: 128 neurons with ReLU activation function
- **Hidden Layer 2**: 128 neurons with ReLU activation function
- **Output Layer**: 1 neuron (Q-value regression output)
- **Total Trainable Parameters**: 17,281
  - FC1: (12 × 128) + 128 bias = 1,664 parameters
  - FC2: (128 × 128) + 128 bias = 16,512 parameters
  - FC3: (128 × 1) + 1 bias = 129 parameters
- **Activation Functions**: ReLU for hidden layers, linear for output
- **Weight Initialization**: Xavier uniform initialization
- **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- **Learning Rate (α)**: 0.001 (Adam optimizer with adaptive learning)
- **Discount Factor (γ)**: 1.0 (undiscounted rewards prioritizing immediate performance)
- **Initial Exploration Rate (ε)**: 0.3 (30% random action probability)
- **Minimum Exploration Rate**: 0.01 (1% residual exploration)
- **Epsilon Decay Rate**: 0.0001 per episode (gradual exploitation increase)
- **Target Episodes**: 15,000 
- **Policy Strategy**: ε-greedy with exponential decay, balancing exploration and exploitation throughout training
- **Staying Alive Reward**: +0.1 for each step maintaining flight safety within operational envelope
- **Successful Landing Reward**: +100.0 for achieving safe landing (MSL < 116m, distance < 100m, lateral deviation < 50m)
- **Terminal Condition Penalty**: -100.0 for any terminal condition violation except successful landings

#### Accelerated Learning Through Early Episode Termination
Episodes are terminated much earlier than actual crashes to dramatically speed up the learning process by maximizing episode turnover rate while maintaining precision requirements.

#### Safety-Critical Terminal Conditions (7 constraints):
1. **Ground contact**: MSL altitude < 116 meters
2. **Approach area**: Distance > 6000 meters
3. **Lateral runway deviation**: |lateral_dev| > 500 meters
4. **Vertical path deviation**: |vertical_dev| > 120 meters
5. **Excessive pitch attitude**: |pitch| > 35 degrees
6. **Excessive bank angle**: |bank| > 40 degrees
7. **Airspeed**: airspeed < 25 or > 120 knots

#### Normalization-Termination Alignment
The normalization bounds are specifically designed to align with the termination conditions. When any state parameter reaches its normalization boundary (±1.0 or 0.0/1.0), the episode terminates. This design choice optimizes the learning process by ensuring the neural network operates within the exact bounds that define operational limits, eliminating the disconnect between feature space and actual constraints.

- **Maximum Simulation Speed**: We use XPLANE's native sim_speed dataref to increase the speed at which the simulation runs at: 8x time acceleration
- **Training Frequency**: One SARSA update per X-Plane physics frame
