from ple.games.flappybird import FlappyBird
from ple import PLE
from DQNAgent import *
import torch
import datetime
import os

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

# #===============================DEBUG BEGIN============================##
# import sys
# sys.path.append("pycharm-debug-py3k.egg")
# import pydevd
#
# pydevd.settrace('127.0.0.1', port=5678, stdoutToServer=True,
# stderrToServer=True)
# #================================DEBUG END=============================##


game = FlappyBird()
p = PLE(game, fps=30, display_screen=False)
p.init()

flappy_agent = DQNAgent(p.getActionSet(), frame_stack=4)

flappy_trainer = Trainer(p, flappy_agent, ReplayMemory, batch_size=32, memory_size=10000)

flappy_trainer.run_experiment(10)

now = datetime.datetime.now()

now_str = now.strftime("%Y-%m-%d-%H-%M")

if not os.path.exists('./models'):
    os.makedirs('./models')

torch.save(flappy_agent.q_network.state_dict(), './models/params_dqn_' + now_str + '.pth')
