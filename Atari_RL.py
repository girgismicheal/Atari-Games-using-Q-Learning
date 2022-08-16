from IPython.display import clear_output
from time import sleep
import gym
import numpy as np
import random


# Setup the game environment
def get_env(env_name):
    """ This function takes the environment name and return the environment after resetting
    input: env_name -> string
    return: env -> the environment object
    """
    env = gym.make(env_name)
    env.reset()  # reset environment to a new, random state
    return env


# build frames of the game till it’s done
def frame_builder(env):
    """  this function take the env and take actions till the game done and return the frames of the game

    Input:
        env -> environment object
    Output:
        frames -> list of dictionaries as each frame has [{action, frame, reward, state},....]
    """
    env.render()
    epochs = 0
    penalties, reward = 0, 0
    frames = []
    done = False

    while not done:
        # automatically selects one random action
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        }
        )
        epochs += 1
    return frames


def print_frames(frames):
    """" this fucntion go pass over the frames to show us each frame and it's info

    Input:
        the frames

    print:
        frame, state, action, and reward
    """
    for i, frame in enumerate(frames):
        # clear_output(wait=True)
        # print(frame['frame'].getvalue())
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)


def q_table_train(env, alpha=0.1, gamma=0.6, epsilon=0.1, decay_over=False, decay_factor=.1):
    """
    This function is for building the  q-table with trained weights and use the decay over

    Input :
        alpha (float)-> the learning rate -> scaler
        gamma (float) -> the discount factor -> scaler
        epsilon (float) ->the epsilon-greedy action selection -> scaler

        decay_over -> Boolen varible
        decay_factor -> float to manage the speed of decaying

    Output :
        q-table (list)
    """
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(1, 100001):  # 100001
        if decay_over and (i % 5000 == 0):
            alpha, gamma, epsilon = alpha * (1 - alpha * decay_factor), gamma * (1 - gamma * decay_factor), epsilon * (
                    1 - epsilon * decay_factor)

        state = env.reset()
        epochs, penalties, reward, = 0, 0, 0
        done = False
        # decay over episode
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")
    return q_table


def model_evaluate(env, q_table):
    """
    the function take the env object and the q-table list to find the AVG_timesteps, and the AVG_penalities

    Input:
        env (object type)
        q_table (list)

    Output:
        frames (list)-> list of frames
        AVG_timesteps (float)-> the average time steps
        AVG_penalities (float)-> the average penalites
    """
    frames = []
    total_epochs, total_penalties = 0, 0
    episodes = 100
    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)
            if reward == -10:
                penalties += 1
            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            }
            )
            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    AVG_timesteps = total_epochs / episodes
    AVG_penalities = total_penalties / episodes
    return frames, AVG_timesteps, AVG_penalities


def train_model(env_name="Taxi-v3", alpha_para=0.1, gamma_para=0.6, epsilon_para=0.1, decay_over=False,
                decay_factor=.1):
    """ the function work to train the model using the parameters plus gaving an option to apply the decay over episodes with a decay factor

    Input:
      env_name (String): the game name
      alpha_para (float), gamma_para (float), epsilon_para (float)

      decay_over (boolean) -> to apply the decay technique or not
      decay_factor (float): due to the decay equation we need the decay_factor, the Equation (parameter*(1-parameter*decay_factor) )

    Output:
      frames (list): list of frames
      AVG_timesteps (float)-> the average time steps
      AVG_penalities (float)-> the average penalites

    """
    env = get_env(env_name)
    # frames= frame_builder(env)
    q_table = q_table_train(env, alpha=alpha_para, gamma=gamma_para, epsilon=epsilon_para, decay_over=decay_over,
                            decay_factor=decay_factor)
    frames, AVG_timesteps, AVG_penalities = model_evaluate(env, q_table)

    return frames, AVG_timesteps, AVG_penalities


def grid_search(env_name="Taxi-v3", parameters={'alpha': [0.9], 'gamma': [0.9], 'epsilon': [.9]}, decay_over=False,
                decay_factor=.1):
    """
    This function try to find the best compination of parmteres with respect to the lowest penalty with minimum timesteps

    Input:
        env_name (string) -> Game name
        parameters (dict) -> Dictionary of lists for each parameter; Example:{'alpha':[0.9],'gamma':[0.9],'epsilon':[.9]}

        decay_over (boolean) -> to apply the decay technique or not
        decay_factor (float) -> due to the decay equation we need the decay_factor, the Equation (parameter*(1-parameter*decay_factor) )

    Output:
        best_params (dict) -> with the best paramters
        best_AVGtime (float) -> the best avarage time
        best_AVGpenalties (float) -> the least penalty value
        best_frame (list)

    """
    best_AVGtime, best_AVGpenalties = 999999, 999999
    best_frame = None
    best_params = {}

    for alpha in parameters['alpha']:
        for gamma in parameters['gamma']:
            for epsilon in parameters['epsilon']:
                frames, AVG_timesteps, AVG_penalities = train_model(env_name, alpha_para=alpha, gamma_para=gamma,
                                                                    epsilon_para=epsilon, decay_over=decay_over,
                                                                    decay_factor=decay_factor)
                if AVG_penalities <= best_AVGpenalties:
                    if AVG_timesteps <= best_AVGtime:
                        best_AVGtime, best_AVGpenalties = AVG_timesteps, AVG_penalities
                        best_params = {'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon}
                        best_frame = frames

    return best_params, best_AVGtime, best_AVGpenalties, best_frame
