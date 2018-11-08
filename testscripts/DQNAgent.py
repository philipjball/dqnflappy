import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as torch

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
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

	def __init__(self, num_actions = 1, input_height=512, input_width=288):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.conv3 = nn.Container(32, 32, kernel_size=5, stride=2)
		self.linear1 = nn.Linear(self.calculate_final_size(input_height, input_width) , 64)
		self.linear2 = nn.Linear(64, num_actions)
	
	@staticmethod
	def calculate_conv_out(input_height, input_width, kernel_size, stride):
		convcalc lambda x, ks, s: (x - ks)/s + 1
		h_out = convcalc(input_height, kernel_size, stride)
		w_out = convcalc(input_width, kernel_size, stride)
		assert h_out.is_integer(), "h_out is not an integer, is in fact %r" % h_out
		assert w_out.is_integer(), "w_out is not an integer, is in fact %r" % w_out
		return int(h_out), int(w_out)

	def calculate_final_size(input_height, input_width):
		ho1, wo1 = self.calculate_conv_out(input_height, input_width, 5, 2)
		ho2, wo2 = self.calculate_conv_out(ho1, wo1, 5, 2)
		ho3, wo3 = self.calculate_conv_out(ho2, wo2, 5, 2)
		return int(ho3 * wo3 * 32)


	def forward(x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		return self.linear2(linear1(x.view(x.size(0), -1)))

class Trainer(object):

	def __init__(self, )
    

