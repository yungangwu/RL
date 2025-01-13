import datetime
from distutils.command.config import config
import pathlib
import sys

# sys.path.append('/home/yg/code/Muzero/muzero/games')
# from DDZ.utils import *
# from DDZ.state import State

# from env.env import Env as ddz
import random
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

import gym
import numpy
import torch


from abstract_game import AbstractGame

try:
    import cv2
except ModuleNotFoundError:
    raise ModuleNotFoundError('Please run "pip install gym[atari]"')

seed = 42
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     numpy.random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(seed)

class MuZeroConfig:
    def __init__(self,):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (1,1,790)  # Dimensions of the model observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shapes = [(1,1,790),(1,1,901),(1,1,901)] # The observation shapes of Multiplayers' game.
        self.action_space = list(range(27472))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation
        self.num_players = 3 # Number of players in the game.
        self.reward = [0, 1, 1]

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "self"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_self_plays = 4
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 40  # Maximum number of moves if game is not finished before
        self.num_simulations = 30  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
        self.k_sample = 100

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = "resnet"  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 8  # Number of channels in the ResNet
        self.reduced_channels_reward = 4  # Number of channels in reward head
        self.reduced_channels_value = 4  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 10
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = int(1e10)  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 16  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 50  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 350e3



        ### Replay Buffer
        self.replay_buffer_size = int(500)  # Number of self-play games to keep in the replay buffers
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 10  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        if seed:
            print('set_seed------------')
            set_seed(seed)
        self.env = rlDdz(seed)

    def set_agents(self,agents):
        self.env.set_agents(agents)

    def step(self, action, player_id):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action, player_id)

        return observation, reward, done

    def to_play(self):
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        pass

class rlDdz():
    def __init__(self, seed) -> None:
        self.env = rlcard.make('doudizhu', config={'seed':seed,})
        self.reward = [0,1,1]

    def set_agents(self, agents):
        self.env.set_agents(agents)

    def to_play(self,):
        return self.env.get_player_id()

    def legal_actions(self):
        player_id = self.env.get_player_id()
        state = self.env.get_state(player_id)

        return list(state['legal_actions'].keys())

    def reset(self):
        state, player = self.env.reset()
        # print(state)
        obs = state['obs']
        obs = numpy.array(obs).reshape(1,1,len(obs))
        return obs

    def step(self, action, player_id):
        # print('agents',self.agents[0])
        state, next_player = self.env.step(action, raw_action=False)
        # print('current_hand----:',state['raw_obs']['current_hand'])
        obs = state['obs']
        # if len(obs) < 901:
        #     obs = numpy.pad(obs,(0,901-len(obs)),'constant',constant_values=(0,0))
        # print(len(obs))
        obs = numpy.array(obs).reshape(1,1,len(obs))
        # player_id = self.env.get_player_id()
        reward = self.env.get_payoffs()
        # print('reward----',reward)
        # reward = reward[player_id]
        # print('player_id------',player_id)
        # print(reward)
        done = self.env.is_over()
        if done:
            print(self.env.get_perfect_information())
        # print('reward----',reward)
        # print('done-------',done)

        return obs, reward, done
