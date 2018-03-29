# Deep Q learning using TensorFlow and TFLearn

The implementation is based on:
  - [Human-level control through deep reinforcement learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.htm)
  - [CartPole problem](https://gym.openai.com/envs/CartPole-v0/)

During training the cart learns how to balance the inverted pendulum using only visual data and no initial training dataset.
![Summary](https://github.com/peternagy1332/deep-q-learning/blob/master/assets/summary.png?raw=true)

## Getting started
Since OpenAI Gym runs only on Linux distributions, this project is not compatible with Windows. Despite this, if you modify the environment wrapper, it could run on Windows machines too.

1. Install Python 3.6.4
1. Run the following command sequence:

```bash
python3 -m venv p36
source p36/bin/activate
git clone https://github.com/peternagy1332/deep-q-learning.git
cd deep-q-learning
pip install -r requirements.txt
python main.py
```
## Used game environments

### CartPoleRawImg-v0
A modified CartPole-v0 implementation that returns frames headlessly: https://github.com/adamtiger/gym.git

[CartPoleRawImg-v0](https://github.com/peternagy1332/deep-q-learning/blob/master/assets/CartPoleRawImg-v0/original.png?raw=true)

### CartPole-v0
The original OpenAI Gym CartPole-v0 game environment.

[CartPoleRawImg-v0](https://github.com/peternagy1332/deep-q-learning/blob/master/assets/CartPoleRawImg-v0/original.png?raw=true)

### Hopper-v2

[Hopper-v2](https://github.com/peternagy1332/deep-q-learning/blob/master/assets/Hopper-v2/original.png?raw=true)

### MountainCar-v0

[MountainCar-v0](https://github.com/peternagy1332/deep-q-learning/blob/master/assets/MountainCar-v0/original.png?raw=true)