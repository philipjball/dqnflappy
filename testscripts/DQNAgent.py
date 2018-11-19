import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import random
from torchvision import transforms
from collections import namedtuple
from PIL import Image
from tensorboardX import SummaryWriter
from collections import deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


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

    def __init__(self, action_set, frame_stack=4, input_height=84, input_width=84):
        super(DQN, self).__init__()
        num_actions = len(action_set)
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(self.calculate_final_size(input_height, input_width), 512)
        self.linear2 = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        return self.linear2(x)

    @staticmethod
    def calculate_conv_out(input_height, input_width, kernel_size, stride):
        def convcalc(x, ks, s): return (x - ks) / s + 1
        h_out = convcalc(input_height, kernel_size, stride)
        w_out = convcalc(input_width, kernel_size, stride)
        assert h_out.is_integer(), "h_out is not an integer, is in fact %r" % h_out
        assert w_out.is_integer(), "w_out is not an integer, is in fact %r" % w_out
        return int(h_out), int(w_out)

    def calculate_final_size(self, input_height, input_width):
        ho1, wo1 = self.calculate_conv_out(input_height, input_width, 8, 4)
        ho2, wo2 = self.calculate_conv_out(ho1, wo1, 4, 2)
        ho3, wo3 = self.calculate_conv_out(ho2, wo2, 3, 1)
        return int(ho3 * wo3 * 64)


class DQNAgent(object):

    def __init__(self, action_set, frame_stack=4, input_height=84, input_width=84):
        self.frame_stack = frame_stack
        self.q_network = DQN(action_set, frame_stack=frame_stack, input_height=input_height,
                             input_width=input_width).to(device)
        self.q_target = DQN(action_set, frame_stack=frame_stack, input_height=input_height,
                            input_width=input_width).to(device)
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
            argmax = torch.argmax(self.q_network(in_frame))
            return self.action_set[argmax]


class DQNLoss(nn.Module):

    def __init__(self, q_network, q_target, action_set, gamma=0.9):
        super(DQNLoss, self).__init__()
        self.q_network = q_network
        self.q_target = q_target
        self.action_set = action_set
        self.gamma = gamma
        self.loss = nn.SmoothL1Loss()

    def forward(self, transition_in):
        states, actions, next_states, rewards, done = zip(*transition_in)     # https://stackoverflow.com/questions/7558908/unpacking-a-list-tuple-of-pairs-into-two-lists-tuples
        states = torch.FloatTensor(states).float().to(device) / 255
        actions_index = torch.LongTensor([self.action_set.index(action) for action in actions]).to(device)
        next_states = torch.FloatTensor(next_states).float().to(device) / 255
        rewards = torch.FloatTensor(rewards).to(device)
        done = torch.FloatTensor(done).to(device)
        pred_return_all = self.q_network(states)
        pred_return = pred_return_all.gather(1, actions_index.unsqueeze(1)).squeeze()           # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
        one_step_return = rewards + self.gamma * self.q_target(next_states).detach().max(1)[0] * (1 - done)
        return self.loss(pred_return, one_step_return)


class Runner(object):
    """Runs the experiments (training and testing inherit from this)"""
    def __init__(self, env, agent, downscale=84, max_ep_steps=1000000):
        self.env = env
        self.agent = agent
        self.transformer = transforms.Compose([transforms.Resize([downscale,downscale])])
        self.total_steps = 0
        self.max_ep_steps = max_ep_steps
        frame_stack = self.agent.frame_stack
        self.frame_stacker = deque(maxlen=frame_stack)

    def preprocess_image(self, input_image):
        """Does a few things:
        1. Reshape image to 84x84
        2. Permute images to the PyTorch ordering (CxHxW)
        3. Convert to PyTorch Tensor
        """
        input_image = Image.fromarray(input_image)
        input_image = self.transformer(input_image)
        input_image = np.array(input_image, dtype=np.uint8).T
        return input_image

    def episode(self):
        """Runs an episode"""
        raise NotImplementedError

    def run_experiment(self):
        """Runs a number of epsiodes"""
        raise NotImplementedError

    def get_recent_states(self):
        assert len(self.frame_stacker) == self.agent.frame_stack, "Not filled enough frames for stacking!"
        return np.array(self.frame_stacker)


class Trainer(Runner):
    # TODO: Frameskip
    def __init__(self, env, agent, loss_func, memory_func, batch_size=32, downscale=84,
                 memory_size=500000, max_ep_steps=1000000, reset_target=10000, final_exp_frame=1000000, gamma=0.9,
                 optimizer=optim.Adam):
        super(Trainer, self).__init__(env, agent, downscale, max_ep_steps)
        self.loss = loss_func(self.agent.q_network, self.agent.q_target, self.agent.action_set, gamma)
        self.memory = memory_func(memory_size)
        self.optimizer = optimizer(self.agent.q_network.parameters(), lr=1e-4)
        self.batch_size = batch_size
        self.reset_target = reset_target
        self.final_exp_frame = final_exp_frame  # Final frame for exploration (whereby we go to eps = 0.1 henceforth)
        self.reward_per_ep = []
        self.tb_writer = SummaryWriter()

    def episode(self):
        steps = 0
        # Do resets
        self.env.reset_game()
        self.frame_stacker.clear()
        rewards = []
        # Start training episode
        state = self.preprocess_image(self.env.getScreenGrayscale())
        self.frame_stacker.append(state)
        while self.env.game_over() is False and steps < self.max_ep_steps:
            # Need to fill frame stacker
            if steps < self.agent.frame_stack:
                action = None
            else:
                psi_state = self.get_recent_states()
                psi_state_tensor = torch.FloatTensor(psi_state).unsqueeze(0).float().to(device) / 255
                action = self.agent.get_action(psi_state_tensor)
            reward = self.env.act(action)
            reward = np.clip(reward, -1.0, 1.0)
            rewards.append(reward)
            state = self.preprocess_image(self.env.getScreenGrayscale())
            self.frame_stacker.append(state)
            if steps > self.agent.frame_stack:
                psi_next_state = self.get_recent_states()
                self.memory.push(psi_state, action, psi_next_state, reward, self.env.game_over())
            self.total_steps += 1
            steps += 1
            self.set_eps()
            if self.total_steps <= self.batch_size + self.agent.frame_stack:
                # if there's not enough samples accumulated, then don't backprop
                continue
            trans_batch = self.memory.sample(self.batch_size)
            loss = self.loss(trans_batch)

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.total_steps % self.reset_target == 0:   # sync up the target and
                print("Updating Target")
                self.agent.update_target()

            # prints and logs
            self.tb_writer.add_scalar('DQN_Flappy/loss', float(loss), self.total_steps)
            self.tb_writer.add_scalar('DQN_Flappy/reward_per_ep', np.mean(self.reward_per_ep), self.total_steps)
            self.tb_writer.add_scalar('DQN_Flappy/epsilon', self.agent.eps, self.total_steps)
            if self.total_steps % 1000 == 0:
                print('Loss at %d steps is %.5f' % (self.total_steps, float(loss)))
                print('Mean reward per episode is:', np.mean(self.reward_per_ep))
                print('epsilon is', self.agent.eps)
                self.reward_per_ep = []

        self.reward_per_ep.append(np.sum(rewards))

    # Function to decrease exploration as we increase steps (copying the DeepMind paper)
    def set_eps(self):
        self.agent.eps = np.max([0.1, (0.1 + 0.9 * (1 - self.total_steps/self.final_exp_frame))])

    def run_experiment(self, num_episodes=1000):
        print('Beginning Training...')
        for i in range(num_episodes):
            self.episode()


class Tester(Runner):

    def __init__(self, agent, env, downscale):
        super(Tester, self).__init__(env, agent, downscale)

    def episode(self):
        steps = 0
        # Do resets
        self.env.reset_game()
        self.frame_stacker.reset()
        rewards = []
        # Start testing
        state = self.preprocess_image(self.env.getScreenGrayscale())
        self.frame_stacker.push(state)
        while self.env.game_over() is False and steps < self.max_ep_steps:
            # Need to fill frame stacker
            if steps < self.frame_stacker.k:
                action = None
            else:
                psi_state = self.frame_stacker.get()
                action = self.agent.get_action(psi_state)
            reward = self.env.act(action)
            rewards.append(reward)
            state = self.preprocess_image(self.env.getScreenGrayscale())
            self.frame_stacker.push(state)
            self.total_steps += 1
            steps += 1
        print('This episode had %.2f reward' % (np.sum(rewards)))

    def run_experiment(self, num_episodes=1000):
        print('Beginning Testing...')
        for i in range(num_episodes):
            self.episode()
