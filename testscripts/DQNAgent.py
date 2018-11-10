import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        # Stop appending when we reach capacity
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # Modulo to wrap around when we reach the end of the list
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size, False)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, action_set, input_height=512, input_width=288):
        super(DQN, self).__init__()
        num_actions = len(action_set)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.linear1 = nn.Linear(self.calculate_final_size(input_height, input_width), 64)
        self.linear2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.linear2(self.linear1(x.view(x.size(0), -1)))

    @staticmethod
    def calculate_conv_out(input_height, input_width, kernel_size, stride):
        def convcalc(x, ks, s): return (x - ks) / s + 1
        h_out = convcalc(input_height, kernel_size, stride)
        w_out = convcalc(input_width, kernel_size, stride)
        assert h_out.is_integer(), "h_out is not an integer, is in fact %r" % h_out
        assert w_out.is_integer(), "w_out is not an integer, is in fact %r" % w_out
        return int(h_out), int(w_out)

    def calculate_final_size(self, input_height, input_width):
        ho1, wo1 = self.calculate_conv_out(input_height, input_width, 4, 2)
        ho2, wo2 = self.calculate_conv_out(ho1, wo1, 5, 2)
        ho3, wo3 = self.calculate_conv_out(ho2, wo2, 4, 2)
        return int(ho3 * wo3 * 32)


class DQNAgent(object):

    def __init__(self, action_set):
        self.q_network = DQN(action_set)
        self.q_target = DQN(action_set)
        self.eps = 1.0
        self.action_set = action_set

    def update_target(self):
        self.q_target.load_state_dict(self.q_network.state_dict())

    def get_action(self, in_frame):
        # eps-greedy exploration
        # if rand number is greater than eps, then explore
        if np.random.rand() < self.eps:
            return np.random.choice(self.action_set)
        else:
            argmax = np.argmax(self.q_network(in_frame))
            return self.action_set[argmax]


class DQNLoss(nn.Module):

    def __init__(self, q_network, q_target, action_set, gamma=0.9):
        super(DQNLoss, self).__init__()
        self.q_network = q_network
        self.q_target = q_target
        self.action_set = action_set
        self.gamma = gamma

    def forward(self, transition_in, game_over):
        state, action, next_state, reward = transition_in
        pred_return = self.q_network(state)[self.action_set.index(action)]
        if not game_over:
            one_step_return = reward + self.gamma * torch.max(self.q_target(next_state))
        else:
            one_step_return = reward
        return F.mse_loss(pred_return, one_step_return)


class Trainer(object):

    def __init__(self, env, agent, loss_func, memory_func, batch_size=32, memory_size=500000, max_ep_steps=1000000,
                 reset_target=10000, gamma=0.9, optimizer=optim.Adam):
        self.env = env
        self.agent = agent
        self.loss = loss_func(self.agent.q_network, self.agent.q_target, self.agent.action_set, gamma)
        self.memory = memory_func(memory_size)
        self.optimizer = optimizer(self.agent.q_network.params)
        self.max_ep_steps = max_ep_steps
        self.batch_size = batch_size
        self.reset_target = reset_target
        self.total_steps = 0

    def episode(self):
        # TODO: Figure out how to get stacked input from environment
        steps = 0
        state = self.env.getScreenRGB()
        while self.env.game_over() is False and steps < self.max_ep_steps:
            action = self.agent.q_network(state)
            reward = self.env.act(action)
            next_state = self.env.getScreenRGB()
            self.memory.push(state, action, next_state, reward)
            state = next_state
            trans_batch = self.memory.sample(self.batch_size)
            loss = self.loss(trans_batch, self.env.game_over())
            loss.backward()
            self.optimizer.step()
            self.total_steps += 1
            steps += 1
            if self.total_steps % self.reset_target == 0:
                self.agent.update_target()

    def run_training(self, num_episodes=1000):
        for i in range(num_episodes):
            self.episode()