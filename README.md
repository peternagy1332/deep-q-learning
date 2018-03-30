# Deep Q learning using TensorFlow and TFLearn

## Summary
During training the agent learns how to solve an optimization problem using only visual data and no initial training dataset.

The implementation is based on the following paper: [Human-level control through deep reinforcement learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.htm)

![Summary](https://github.com/peternagy1332/deep-q-learning/blob/master/assets/CartPole-v0/summary.png?raw=true)

## Getting started
Since OpenAI Gym runs only on Linux distributions, this project is not compatible with Windows. Despite this, if you modify the environment wrapper, it could run on Windows machines too.

1. Install Python 3.5.2
1. Run the following command sequence:

```bash
python3 -m venv p35
source p35/bin/activate
git clone https://github.com/peternagy1332/deep-q-learning.git
cd deep-q-learning
pip install -r requirements.txt
python main.py [-d default_configs/YOUR_CONFIG.yaml] [-m models/YOUR_MODEL] [-t]
```
Where [...] means ... is an optional argument.

### Command line arguments
```bash
$ python main.py -h

usage: main.py [-h] [-m MODEL_DIR] [-t] [-d DEFAULT_CONFIG]

Deep Q Learning

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_DIR, --model-dir MODEL_DIR
                        Path to a new or existing model directory.
                        Default: models/model--YY-MM-DD--HH:MM:SS
  -t, --train-model     If true, model will be trained. Otherwise, it will be
                        evaluated.
                        Default: True
  -d DEFAULT_CONFIG, --default-config DEFAULT_CONFIG
                        The default config to use when creating a new model.
                        Default: default_configs/CartPole-v0.yaml
```

## Used game environments

### CartPole-v0
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

#### Original input
![CartPole-v0](https://github.com/peternagy1332/deep-q-learning/blob/master/assets/CartPole-v0/original.png?raw=true)

#### Evaluation
![CartPole-v0-eval](https://github.com/peternagy1332/deep-q-learning/blob/master/models/model-CartPole-v0/eval.png?raw=true)

### CartPoleRawImg-v0
A modified CartPole-v0 implementation that returns frames headlessly: https://github.com/adamtiger/gym.git

#### Original input
![CartPoleRawImg-v0](https://github.com/peternagy1332/deep-q-learning/blob/master/assets/CartPoleRawImg-v0/original.png?raw=true)

##### Evaluation
![CartPoleRawImg-v0-eval](https://github.com/peternagy1332/deep-q-learning/blob/master/models/model-CartPoleRawImg-v0/eval.png?raw=true)

### Acrobot-v1
The acrobot system includes two joints and two links, where the joint between the two links is actuated. Initially, the links are hanging downwards, and the goal is to swing the end of the lower link up to a given height.

#### Original input
![Acrobot-v1](https://github.com/peternagy1332/deep-q-learning/blob/master/assets/Acrobot-v1/original.png?raw=true)

##### Evaluation
Training in progress...
![Acrobot-v1-eval](https://github.com/peternagy1332/deep-q-learning/blob/master/models/model-Acrobot-v1/eval.png?raw=true)

### MountainCar-v0
A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

#### Original input
![MountainCar-v0](https://github.com/peternagy1332/deep-q-learning/blob/master/assets/MountainCar-v0/original.png?raw=true)

##### Evaluation
Training in progress...
![MountainCar-v0-eval](https://github.com/peternagy1332/deep-q-learning/blob/master/models/model-MountainCar-v0/eval.png?raw=true)
