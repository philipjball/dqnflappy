from ple.games.flappybird import FlappyBird
from ple import PLE
from DQNAgent import *
import os
#
# os.putenv('SDL_VIDEODRIVER', 'fbcon')
# os.environ["SDL_VIDEODRIVER"] = "dummy"

# import sys
# sys.path.append("pycharm-debug-py3k.egg")
# import pydevd
#
# pydevd.settrace('127.0.0.1', port=5678, stdoutToServer=True,
# stderrToServer=True)

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
p.init()

flappy_agent = DQNAgent(p.getActionSet(), frame_stack=4)

flappy_trainer = Trainer(p, flappy_agent, DQNLoss, ReplayMemory, batch_size=32)

flappy_trainer.run_training(10)