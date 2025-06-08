import numpy as np
import torch as T
import gym
from matplotlib import pyplot as plt
from matplotlib import get_backend
from dqn import DQAgent
from IPython import display
from itertools import count

# set up matplotlib
is_ipython = 'inline' in get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_durations = []

def main():
    """ let us define the hyperparameters for the DQN agent """
    #learning rate alpha for DNN
    alpha = 1e-4
    #RL discount factor gamma
    gamma = 0.75
    #polyak averaging factor tau
    tau = 0.005
    #observation space dimensions
    gym_env = gym.make('CartPole-v0')
    input_dims = gym_env.observation_space.shape
    #number of actions
    n_actions = gym_env.action_space.n
    #epsilon greedy exploration factor
    eps = 0.9
    #epsilon decay factor
    eps_dec = 1000
    #minimum epsilon value
    eps_min = 0.1
    #batch size for training
    batch_size = 128
    #size of replay memory
    mem_size = 1000000
    #number of steps before replacing target network
    replace = 1000
    print(f'Input dimensions: {input_dims}, Number of actions: {n_actions}')
    #squee
    #let us define the DQN agent
    agent = DQAgent(alpha, gamma, tau, input_dims, n_actions, eps, eps_dec, eps_min, batch_size, mem_size)

    n_games = 500
    score_history = []
    best_score = gym_env.reward_range[0]
    learn_iters = 0

    for i in range(n_games):
        observation = gym_env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        done = False
        score = 0
        for t in count():
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = gym_env.step(action)
            done = terminated or truncated
            score += reward
            mem_size = agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            agent.soft_update_target_network()
            observation = observation_
            learn_iters += 1

            if done:
                episode_durations.append(t + 1)
                print(f'Episode {i+1} finished after {t + 1} timesteps')
                plot_durations() 
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_model()
            print(f'New best score: {best_score:.2f}, saving model...')

        print(f'Episode: {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Learning Steps: {learn_iters}')

    x = [i+1 for i in range(len(score_history))]
    plt.plot(x, score_history)
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.title('DQN on CartPole-v0')
    plt.show()

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()    

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = T.tensor(episode_durations, dtype=T.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = T.cat((T.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

if __name__ == '__main__':
    main()

    


