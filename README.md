# Flappy Bird DQN

This repo implements DQN (from scratch) and learns to play Flappy Bird, specifically the PyGame/PLE implementation

In its current version, I get the following performance averaged over 20 episodes:

| Algorithm | Performance |
| :----:       | :----:         |
| DQN Vanilla  | 119.1   |

NB: The default arguments are the hyperparameters used.

## Requirements

* Python 3.6
* NumPy 1.15
* PyTorch 0.4.1
* OpenCV-Python 3.4.3
* TensorboardX 1.4
* PyGame 1.9.4
* [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment) 0.0.1

## Usage

To visualise the agent with pre-trained weights, simply type:
```bash
python runFlappy.py
```
The various options are as follows:
```bash
python runFlappy.py

## High Level Settings
--mode 'test'                               # one of 'train' or 'test'
--testfile './models/trained_params.pth     # location of pretrained model (if 'test' is selected)
--slow False                                # run at native 30 FPS (seems less stable)

## Training Settings
--frame_skip 3                              # how many frames will be skipped and the same action will be applied
--reward_shaping True                       # include a living reward
--frame_stack 4                             # how many frames to stack
--gamma 0.9                                 # future return discounting
--batch_size 32                             # size of batch from replay memory buffer
--memory_size 100000                        # size of the entire replay memory buffer
--max_ep_steps 1000000                      # how many steps we can spend in a single episode
--reset_target 10000                        # how many steps before syncing the target q-network
--final_exp_frame 500000                    # how many steps before we settle on the final exploration value
--save_freq 10000                           # how many steps between saving model parameters
--num_episodes 100000000                    # how many episodes to run in total (basically infinite)
--num_samples_pre 3000                      # how many samples under a random policy to initially load into the replay memory
```


## TODO:
* Implement some of Rainbow (see spinning up)
* Fork and add more
    * Games (Pong, OpenAI Gym)
    * Algorithms (A2C, PPO, etc.)
    

