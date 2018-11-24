from ple.games.flappybird import FlappyBird
from ple import PLE
from DQNAgent import *
import argparse
import os


def setup_env_agent(display_screen, frame_skip, force_fps, reward_shaping, frame_stack, train):
    game = FlappyBird()
    ple_flappy = PLE(game, fps=30, display_screen=display_screen, frame_skip=frame_skip, force_fps=force_fps)
    if reward_shaping and train:
        z = ple_flappy.game.rewards
        z['tick'] = 0.1
        ple_flappy.game.adjustRewards(z)
    ple_flappy.init()
    agent = DQNAgent(ple_flappy.getActionSet(), frame_stack=frame_stack)

    return ple_flappy, agent


def check_train_test(value):
    if value in ['train', 'test']:
        return value
    else:
        raise argparse.ArgumentTypeError("%s is not 'train' or 'test'" % value)


def main(args):

    if args.mode == 'train':
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        display_screen = False
        force_fps = True
        env, flappy_agent = setup_env_agent(display_screen=display_screen, frame_skip=args.frame_skip,
                                            force_fps=force_fps, reward_shaping=args.reward_shaping,
                                            frame_stack=args.frame_stack, train=True)
        flappy_runner = Trainer(env, flappy_agent, ReplayMemory, batch_size=args.batch_size,
                                memory_size=args.memory_size, final_exp_frame=args.final_exp_frame,
                                save_freq=args.save_freq, reset_target=args.reset_target,
                                max_ep_steps=args.max_ep_steps, gamma=args.gamma)
    else:
        display_screen = True
        force_fps = not args.slow
        env, flappy_agent = setup_env_agent(display_screen=display_screen, frame_skip=args.frame_skip,
                                            force_fps=force_fps, reward_shaping=args.reward_shaping,
                                            frame_stack=args.frame_stack, train=False)
        flappy_agent.eps = 0.0
        flappy_runner = Tester(env, flappy_agent, 84)
        flappy_runner.load_model(args.testfile)

    flappy_runner.run_experiment(args.num_episodes)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run DQN on Flappy Bird, training/testing.')
    parser.add_argument('--mode', type=check_train_test, nargs=1, default='test',
                        help='set to train or test')
    parser.add_argument('--testfile', type=str, nargs=1, default='./models/trained_params.pth',
                        help='path of the trained model')
    parser.add_argument('--frame_skip', type=int, nargs=1, default=3,
                        help='how many frames to skip between actions (and apply the same action)')
    parser.add_argument('--reward_shaping', type=bool, nargs=1, default=True,
                        help='incorporate a reward for living')
    parser.add_argument('--frame_stack', type=int, nargs=1, default=4,
                        help='how many frames to stack together as input to DQN')
    parser.add_argument('--gamma', type=float, nargs=1, default=0.9,
                        help='future return discounting parameter')
    parser.add_argument('--batch_size', type=int, nargs=1, default=32,
                        help='batch size from replay memory buffer')
    parser.add_argument('--memory_size', type=int, nargs=1, default=100000,
                        help='size of replay memory buffer')
    parser.add_argument('--max_ep_steps', type=int, nargs=1, default=1000000,
                        help='max number of steps per episode')
    parser.add_argument('--reset_target', type=int, nargs=1, default=10000,
                        help='steps before syncing the target network with the current DQN')
    parser.add_argument('--final_exp_frame', type=int, nargs=1, default=500000,
                        help='frame at which exploration hits the baseline (i.e., 0.01)')
    parser.add_argument('--save_freq', type=int, nargs=1, default=10000,
                        help='how often to save the models')
    parser.add_argument('--num_episodes', type=int, nargs=1, default=100000000,
                        help='how many episodes to run')
    parser.add_argument('--num_samples_pre', type=int, nargs=1, default=3000,
                        help='num of samples with a random policy to seed the replay memory buffer')
    parser.add_argument('--slow', type=bool, nargs=1, default=False,
                        help='run the game in the native FPS (not as fast as possible)')
    arguments = parser.parse_args()

    main(arguments)
