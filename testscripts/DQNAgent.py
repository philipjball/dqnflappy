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

    def __init__(self, action_set=[None, None], input_height=512, input_width=288):
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
        convcalc = lambda x, ks, s: (x - ks) / s + 1
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

# class Trainer(object):

# 	def __init__(self, ):
