# Atari-Games-using-Q-Learning

## Overview
In this project, I'm dealing with multiple environments from the GYM library and trying to apply the Reinforcement Learning technique to optimize the agent actions.
<br>
![RL_AgentModel](Image/RL_AgentModel.jpg)
<br>
it's easier to work with modularized code, as it's simple to use as we would see later
<br> So, the project is divided into three modules:
1) Build a modular code consisting of functions that can be used in multiple environments.
2) Tune alpha, gamma, and/or epsilon using decay over episodes.
3) Implement a grid search to discover the best hyperparameters.


## Table of Contents
- [Usage](#Usage)
  - [Install](#Install)
  - [Import the project file](#Import-the-project-file)
  - [Train on the environment](#Train-on-the-environment)
  - [Train and Evaluate](#Train-and-Evaluate)
    - [Train and Evaluate on the Taxi-v3 environment](#Train-and-Evaluate-on-the-Taxi-v3-environment)
    - [Train and Evaluate on the FrozenLake-v1 environment](#Train-and-Evaluate-on-the-FrozenLake-v1-environment)
  - [Tuning the Parameters using Decay Over Episodes Technique](#Tuning-the-Parameters-using-Decay-Over-Episodes-Technique)
    - [Tuning on the Taxi-v3 environment](#Tuning-on-the-Taxi-v3-environment)
    - [Tuning on the FrozenLake-v1 environment](#Tuning-on-the-FrozenLake-v1-environment)
  - [Use the Grid Search](#Use-the-Grid-Search)
    - [On the Taxi-v3 environment](#On-the-Taxi-v3-environment)
    - [On the FrozenLake-v1 environment](#On-the-FrozenLake-v1-environment)

## Usage
### Install 
I'm using some libraries in the code that should be installed at the beginning by running those commands:
```python
!pip install cmake 'gym[atari]' scipy
!pip install gym[atari]
!pip install autorom[accept-rom-license]
!pip install gym[atari,accept-rom-license]==0.21.0
```
### Import the project file
```Python
from Atari_RL import *
```

### Train on the environment
All you need to train the model on an environment is just pass the environment's name to the train model function.
#### Train on the Taxi-v3 environment
```python
env_name = 'Taxi-v3'
frames, AVG_timesteps, AVG_penalities = train_model(env_name)
print(f"Average timesteps per episode: {AVG_timesteps}")
print(f"Average penalties per episode: {AVG_penalities}")

"""" Output:
Episode: 100000
Training finished.
Results after 100 episodes:
Average timesteps per episode: 20.88
Average penalties per episode: 0.0
""""
```

### Train and Evaluate
#### Train and Evaluate on the Taxi-v3 environment
```python
# Specify the game required
env_name = 'Taxi-v3'
# Return the game environment as an object
env = get_env(env_name)
# Build the Q-Table just specify the learning parameters
q_table=q_table_train(env,alpha =.1,gamma = .6,epsilon = .9)
# Evaluate the model by returning the time and penalties
frames, AVG_timesteps, AVG_penalities= model_evaluate(env, q_table)
# Visualize the game frame by frame
print_frames(frames)
# print the model Average timesteps and Average penalties
print(f"Average timesteps per episode: {AVG_timesteps}")
print(f"Average penalties per episode: {AVG_penalities}")
```
![image](https://drive.google.com/uc?export=view&id=1JbugSE2wC18DotytdMyA4Pmr1Q55OOSD)
#### Train and Evaluate on the FrozenLake-v1 environment
```python
# Specify the game required
env_name = 'FrozenLake-v1'
# Return the game environment as an object
env = get_env(env_name)
# Build the Q-Table just specify the learning parameters
q_table=q_table_train(env,alpha =.1,gamma = .6,epsilon = .9)
# Evaluate the model by returning the time and penalties
frames, AVG_timesteps, AVG_penalities= model_evaluate(env, q_table)
# Visualize the game frame by frame
print_frames(frames)
# print the model Average timesteps and Average penalties
print(f"Average timesteps per episode: {AVG_timesteps}")
print(f"Average penalties per episode: {AVG_penalities}")
```
![image](https://drive.google.com/uc?export=view&id=1Fm7yM5W32CfrSZSCvIGdqrAytuY4Ocpb)


### Tuning the Parameters using Decay Over Episodes Technique
Also, built a function to train and evaluate the model using the decay over episodes technique using this equation: **parameter = parameter\*(1-parameter \* decay_factor)**
```python
# The hyperparameter
alpha = 0.1
gamma = 0.9
epsilon = 0.9
# Apply the decay over technique with decay factor .1
decay_over = True
decay_factor= .1
```
#### Tuning on the Taxi-v3 environment
```python
env_name = 'Taxi-v3'
frames, AVG_timesteps, AVG_penalities = train_model(env_name, alpha_para = alpha, gamma_para =gamma, epsilon_para = epsilon,decay_over=decay_over,decay_factor=decay_factor)
print(f"Average timesteps per episode: {AVG_timesteps}")
print(f"Average penalties per episode: {AVG_penalities}")

""" Output:
Episode: 100000
Average timesteps per episode: 20.88
Average penalties per episode: 0.0
"""
```
#### Tuning on the FrozenLake-v1 environment
```python
env_name = 'FrozenLake-v1'
frames,AVG_timesteps, AVG_penalities = train_model(env_name, alpha_para = 0.1, gamma_para = 0.6, epsilon_para = 0.9,decay_over=True,decay_factor=.1)
print(f"Average timesteps per episode: {AVG_timesteps}")
print(f"Average penalties per episode: {AVG_penalities}")

""" Output:
Episode: 100000
Average timesteps per episode: 8.31
Average penalties per episode: 0.0
"""
```

### Use the Grid Search
It's required to implement Grid Search to find the best combinations of hyper parameters values to get the minimum penalty and minimum steptime.

#### On the Taxi-v3 environment
```python
env_name = "Taxi-v3"
params = {'alpha':[0.9,0.6,0.3],'gamma':[0.9,0.6,0.3],'epsilon':[0.9,0.6,0.3]}
best_params1, best_AVGtime1 ,best_AVGpenalties1, best_frame1 = grid_search(env_name=env_name,parameters=params,decay_over=False,decay_factor=.1)
print('Best_parameters:', best_params1)
print('Average timesteps per episode:', best_AVGtime1)
print('Average penalties per episode:', best_AVGpenalties1)

""" Output:
Best_parameters: {'alpha': 0.6, 'gamma': 0.3, 'epsilon': 0.9}
Average timesteps per episode: 12.43
Average penalties per episode: 0.0
"""
```
#### On the FrozenLake-v1 environment
```python
env_name = "FrozenLake-v1"
params = {'alpha':[0.9,0.6,0.3],'gamma':[0.9,0.6,0.3],'epsilon':[0.9,0.6,0.3]}
best_params1, best_AVGtime1 ,best_AVGpenalties1, best_frame1 = grid_search(env_name=env_name,parameters=params,decay_over=False,decay_factor=.1)
print('Best_parameters:', best_params1)
print('Average timesteps per episode:', best_AVGtime1)
print('Average penalties per episode:', best_AVGpenalties1)

""" Output:
Best_parameters: {'alpha': 0.9, 'gamma': 0.9, 'epsilon': 0.9}
Average timesteps per episode: 5.1
Average penalties per episode: 0.0
"""
```