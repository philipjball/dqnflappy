import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class KLastFrames(object):
    """Does the \psi function within the DQN paper with luminance"""
    def __init__(self, k):
        self.k = k
        self.memory = []

    def push(self, new_frame):
        """Adds a new frame to the temporary cache"""
        if not len(self.memory) == self.k:
            self.memory.append(new_frame)
        else:
            self.memory.pop(0)
            self.memory.append(new_frame)
            assert len(self.memory) == self.k, "Memory is the wrong size!"

    def get(self):
        assert len(self.memory) == self.k, "Trying to get %r-frames before full!" % self.k
        return torch.cat(self.memory)

    def reset(self):
        """Resets the frame stacker"""
        self.memory = []


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
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, action_set, frame_stack=4, input_height=512, input_width=288):
        super(DQN, self).__init__()
        num_actions = len(action_set)
        self.conv1 = nn.Conv2d(frame_stack, 16, kernel_size=4, stride=2)
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

    def __init__(self, action_set, frame_stack=4):
        self.frame_stack = frame_stack
        self.q_network = DQN(action_set, frame_stack=frame_stack)
        self.q_target = DQN(action_set, frame_stack=frame_stack)
        self.eps = 1.0
        self.action_set = action_set

    def update_target(self):
        self.q_target.load_state_dict(self.q_network.state_dict())

    def random_action(self):
        return np.random.choice(self.action_set)

    def get_action(self, in_frame):
        # eps-greedy exploration
        # if rand number is greater than eps, then explore
        if np.random.rand() < self.eps:
            return self.random_action()
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
        states, actions, next_states, rewards = zip(*transition_in)     # https://stackoverflow.com/questions/7558908/unpacking-a-list-tuple-of-pairs-into-two-lists-tuples
        pred_return_all = self.q_network(torch.stack(states))
        pred_return = []
        for pred_return_row, action in zip(pred_return_all, actions):
            pred_return.append(pred_return_row[self.action_set.index(action)].unsqueeze_(0))
        pred_return = torch.cat(pred_return)
        if not game_over:
            one_step_return = torch.Tensor(rewards) + self.gamma * torch.max(self.q_target(torch.stack(next_states)), dim=1)[0]
        else:
            one_step_return = torch.Tensor(rewards)
        return F.mse_loss(pred_return, one_step_return)


class Trainer(object):

    def __init__(self, env, agent, loss_func, memory_func, batch_size=32, memory_size=500000, max_ep_steps=1000000,
                 reset_target=10000, final_exp_frame=1000000, gamma=0.9, optimizer=optim.Adam):
        self.env = env
        self.agent = agent
        self.loss = loss_func(self.agent.q_network, self.agent.q_target, self.agent.action_set, gamma)
        self.memory = memory_func(memory_size)
        self.optimizer = optimizer(self.agent.q_network.parameters())
        self.max_ep_steps = max_ep_steps
        self.batch_size = batch_size
        self.reset_target = reset_target
        self.final_exp_frame = final_exp_frame  # Final frame for exploration (whereby we go to eps = 0.1 henceforth)
        self.total_steps = 0
        frame_stack = self.agent.frame_stack
        self.frame_stacker = KLastFrames(frame_stack)

    @staticmethod
    def preprocess_image(input_image):
        """Permute images to the PyTorch ordering (CxWxH)"""
        if len(input_image.shape) != 3:
            return input_image.unsqueeze_(0)
        return input_image.permute([2,0,1])

    def episode(self):
        steps = 0
        # Do resets
        self.env.reset_game()
        self.frame_stacker.reset()
        # Start training
        state = self.preprocess_image(torch.Tensor(self.env.getScreenGrayscale()))
        self.frame_stacker.push(state)
        while self.env.game_over() is False and steps < self.max_ep_steps:
            # Need to fill frame stacker
            if steps < self.frame_stacker.k:
                # TODO: None vs random?
                # action = self.agent.random_action()
                action = None
            else:
                psi_state = self.frame_stacker.get()
                action = self.agent.get_action(psi_state)
            reward = self.env.act(action)
            state = self.preprocess_image(torch.Tensor(self.env.getScreenGrayscale()))
            self.frame_stacker.push(state)
            if steps > self.frame_stacker.k:
                psi_next_state = self.frame_stacker.get()
                self.memory.push(psi_state, action, psi_next_state, reward)
            self.total_steps += 1
            steps += 1
            self.set_eps()
            if self.total_steps <= self.batch_size + self.frame_stacker.k:    # if there's not enough samples accumulated, then don't backprop
                continue
            trans_batch = self.memory.sample(self.batch_size)
            loss = self.loss(trans_batch, self.env.game_over())
            loss.backward()
            self.optimizer.step()
            if self.total_steps % self.reset_target == 0:   # sync up the target and
                print("Updating Target")
                self.agent.update_target()

    # Function to decrease exploration as we increase steps (copying the DeepMind paper)
    def set_eps(self):
        self.agent.eps = np.max([0.1, (0.1 + 0.9 * (1 - self.total_steps/self.final_exp_frame))])

    def run_training(self, num_episodes=1000):
        # TODO: Print Losses every so often
        for i in range(num_episodes):
            print("Starting New Episode")
            self.episode()