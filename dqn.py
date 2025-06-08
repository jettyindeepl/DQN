import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import os

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = T.device("cuda" if T.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',('state','action','reward','next_state','done'))
                         

class ReplayBuffer():

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self,*args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Samples a batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """Returns the current size of internal memory."""
        return len(self.memory)

class DQNetwork(nn.Module):
    def __init__(self, states_dims, action_dims, alpha, fc1_dims=128, fc2_dims=128, ckpt_dir='tmp/dqn'):
        super(DQNetwork, self).__init__()
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.checkpoint_file = os.path.join(ckpt_dir, 'dqn_torch')
        self.dqn = nn.Sequential(nn.Linear(*states_dims, fc1_dims),\
                                 nn.ReLU(),\
                                 nn.Linear(fc1_dims, fc2_dims),\
                                 nn.ReLU(),\
                                 nn.Linear(fc2_dims, action_dims))
        #self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        #self.optimizer = optim.AdamW(self.parameters(), lr=alpha, amsgrad=True)        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        action_values = self.dqn(state)
        return action_values
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

    def set_optimizer(self, optimizer):
        """Sets the optimizer for the network."""
        self.optimizer = optimizer


class DQAgent():
    def __init__(self, alpha, gamma, tau, n_observations, n_actions, eps_max=0.99, eps_dec=1000, eps_min=0.01, batch_size=64, mem_size=1000000):
        self.nactions = n_actions
        self.gamma = gamma
        self.eps_max = eps_max
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size)
        self.tau = tau

        self.q_eval = DQNetwork(n_observations, n_actions, alpha)
        self.q_eval.set_optimizer(optim.AdamW(self.q_eval.parameters(), lr=alpha, amsgrad=True))  # Set the optimizer for the evaluation network
        self.q_target = DQNetwork(n_observations, n_actions, alpha)
        self.q_target.set_optimizer(None)  # Set the optimizer for the target network
        self.q_target.load_state_dict(self.q_eval.state_dict())

        self.steps_done = 0

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition in the replay buffer."""
        reward = T.tensor([reward], dtype=T.float).to(self.q_eval.device)
        self.memory.push(state, action, reward, next_state, done)
        return self.memory.__len__()

    def choose_action(self, observation):

        eps_thr = self.eps_min + (self.eps_max - self.eps_min) * math.exp(-1 * self.steps_done / self.eps_dec)
        #print(f'epsilon threshold: {eps_thr}, steps done: {self.steps_done}')
        if random.random() > eps_thr:
            state = T.tensor(observation, dtype=T.float).to(self.q_eval.device)
            with T.no_grad():
                # Forward pass through the evaluation network to get action values
                # We use no_grad() to avoid tracking gradients for this inference step
                actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = random.randint(0, self.nactions - 1)
        self.steps_done += 1
        return action
    
    def save_model(self):
        print('... saving model ...')
        self.q_eval.save_checkpoint()
        self.q_target.save_checkpoint()

    def load_model(self):
        print('... loading model ...')
        self.q_eval.load_checkpoint()
        self.q_target.load_checkpoint()

    def learn(self):
        if len(self.memory) < self.batch_size:
            print('Not enough samples in memory to learn.')
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_terminal_mask = T.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype= T.bool).to(self.q_eval.device)
        non_terminal_next_states = T.tensor([s for s in batch.next_state if s is not None], dtype=T.float).to(self.q_eval.device)
        state_batch = T.tensor(batch.state, dtype=T.float) # contains the current states in the batch and each state can be multi dimenstionl. It may look like after converting to tensors [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...]
        action_batch = T.tensor(batch.action, dtype=T.int64) #it is acturally index for each of the actions. Say if we have 5 actions and each of the actions is represented by an index from 0 to 4, then action_batch may look like [0, 1, 2, 3, 4] for a batch of size 5.
        reward_batch = T.tensor(batch.reward, dtype=T.float) # it is the reward for each of the actions in the batch. It may look like [1, 0, 1, 0, 1] for a batch of size 5. Reward is the immediate reward received after taking the action in the current state.
        done_batch = T.tensor(batch.done, dtype=T.float) # it is the done flag for each of the actions in the batch. It may look like [False, True, False, True, False] for a batch of size 5.
        next_state_batch = T.tensor(batch.next_state, dtype=T.float) # it is the next state for each of the actions in the batch. It may look like [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], ...] for a batch of size 5.

        # take the states in the current batch and compute the q-values Q(s_t, for each of the actions). Choose the Q value corresponding to the actions taken in the current batch. This gives Q(s_t,a_t) for the transition.
        #state_action_values = self.q_eval.forward(state_batch).gather(1, action_batch)
        state_action_values = self.q_eval.forward(state_batch)
        #print(f'action_batch: {action_batch}') # this will be of shape (batch_size,) because it is just the index of the action taken in the current state. We will use this to index the state_action_values to get the Q value for the action taken in the current state.
        state_action_value = state_action_values.gather(1, action_batch.unsqueeze(1)) # this will give us the Q value for the action taken in the current state. The unsqueeze(1) is used to make the shape of the action_batch compatible with the state_action_values. The shape of state_action_value will be (batch_size, 1).
        #print(f'state_action_value: {state_action_value}') # this will be of shape (batch_size, 1) because we are taking the Q value for the action taken in the current state. The unsqueeze(1) is used to make the shape of the action_batch compatible with the state_action_values.

        # The q values for the next states are computed with the target network. The weights of the target network are updated less often than the evaluation network and this is done to reduce the fluctuations in the Q values.
        # If the evaluation network is used to compute the q values of the next states, then it is like catching up with itself and results in unstable oscillation.
        next_state_values = T.zeros(self.batch_size)#initialise the next state valuse to zero. In the next steps, we will only update the next state values for th non-terminal states and the termainl states will have values of zero.
        with T.no_grad(): #in the computation of the next state values, we do not want to update the weights of the target network. so we use the no_grad() context manager to avoid updating the weights.
            next_state_values = self.q_target.forward(next_state_batch).max(1)[0] # we take the maximum q value for the next states. This is the Q(S_{t+1}, a_{t+1}). We don't care the action taken in the next state because we just want the max over a.
            next_state_values = next_state_values.unsqueeze(1)
            next_state_values = next_state_values * (1 - done_batch.unsqueeze(1)) # we multiply the next state values with the done flag to make sure that the next state values are zero for the terminal states. The unsqueeze(1) is used to make the shape of the done_batch compatible with the next_state_values. The shape of next_state_values will be (batch_size, 1).
        #print(f'next_state_values: {next_state_values}') # this will be of shape (batch_size, 1) because we are taking the maximum Q value for the next states. The unsqueeze(1) is used to make the shape of the next_state_values compatible with the reward_batch.
        reward_batch = reward_batch.unsqueeze(1) # we unsqueeze the reward_batch to make it compatible with the next_state_values. The shape of reward_batch will be (batch_size, 1).
        expected_state_action_values = reward_batch + self.gamma * next_state_values #here we comute r + gamma * max Q(S_{t+1}, a_{t+1}) for the non-terminal states. For the terminal states, we just use the reward as the expected state action value.
        
        #print(f'expected_state_action_values: {expected_state_action_values}') # this will be of shape (batch_size,) because it is the expected state action value for each action in the batch. It is computed as r + gamma * max Q(S_{t+1}, a_{t+1}) for the non-terminal states and just r for the terminal states.

        #compute huber loss
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(state_action_value, expected_state_action_values) # we use the smooth L1 loss (huber loss) to compute the loss between the Q values of the current states and the expected state action values. The unsqueeze(1) is used to make the shape of the expected state action values compatible with the state action values.
        #print(f"State Action Values shape: {state_action_value.shape}")
        #print(f"Expected State Action Values shape: {expected_state_action_values.shape}")
        #print(f"Loss: {loss.item()}")

        self.q_eval.optimizer.zero_grad() # zero the gradients of the evaluation network
        loss.backward() # compute the gradients of the loss with respect to the weights of the evaluation network
        #inplace gradient clipping to avoid exploding gradients
        nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=1.0)

        #old parameter values
        #old_params = {name: param.clone() for name, param in self.q_eval.named_parameters()}
        self.q_eval.optimizer.step() # update the weights of the evaluation network
        #new parameter values
        #new_params = {name: param for name, param in self.q_eval.named_parameters()}
        #print the difference between the old and new parameters
        #print(f'parameeter change: {[(name, (new_params[name] - old_params[name]).norm().item()) for name in new_params.keys()]}') # this will print the difference between the old and new parameters. It is useful to see if the parameters are changing or not.

        # let us do a soft update of the target network. POLYAK AVERAGING
        
    def soft_update_target_network(self):    
        target_network_state_dict = self.q_target.state_dict()
        eval_network_state_dict = self.q_eval.state_dict()
        for key in eval_network_state_dict.keys():
            target_network_state_dict[key] = eval_network_state_dict[key] * self.tau + target_network_state_dict[key] * (1 - self.tau)
        self.q_target.load_state_dict(target_network_state_dict)
 

                                 