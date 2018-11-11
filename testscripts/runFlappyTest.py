from ple.games.flappybird import FlappyBird
from ple import PLE
from DQNAgent import *

import sys
sys.path.append("pycharm-debug-py3k.egg")
import pydevd

pydevd.settrace('127.0.0.1', port=5678, stdoutToServer=True,
stderrToServer=True)

game = FlappyBird()
p = PLE(game, fps=30, display_screen=False)
p.init()

flappy_agent = DQNAgent(p.getActionSet())

flappy_trainer = Trainer(p, flappy_agent, DQNLoss, ReplayMemory)

flappy_trainer.run_training(10)