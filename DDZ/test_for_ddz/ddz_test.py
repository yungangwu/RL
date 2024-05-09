import rlcard
import torch
import pathlib
import numpy
import ray
import models
import math
import ddz
import self_play_test
import time
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, tournament

seed = 42
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     numpy.random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(seed)

config = ddz.MuZeroConfig()
observation_shapes = config.observation_shapes
configs = []
for i in range(config.num_players):
    config = ddz.MuZeroConfig()
    config.observation_shape = observation_shapes[i]
    configs.append(config)

agents = []
for i in range(config.num_players):
    model = models.MuZeroNetwork(configs[i])
    model = model.cuda()
    model.eval()
    agents.append(model)

ray.init(num_gpus=1, ignore_reinit_error=True)


landlord_paths = '/home/yg/code/Muzero/muzero/results/test_ddz/new/model0.checkpoint'
farmer0_paths = '/home/yg/code/Muzero/muzero/results/test_ddz/old/model1.checkpoint'
farmer1_paths = '/home/yg/code/Muzero/muzero/results/test_ddz/old/model2.checkpoint'

models_paths = [landlord_paths,farmer0_paths,farmer1_paths]

for i in range(config.num_players):
    model_path = pathlib.Path(models_paths[i])
    model_checkpoint = torch.load(model_path)
    agents[i].set_weights(model_checkpoint['weights'])

# Manage GPUs
num_gpus = 1.0
if 0 < num_gpus:
    num_gpus_per_worker = num_gpus / (
         config.num_workers
    )
    if 1 < num_gpus_per_worker:
        num_gpus_per_worker = math.floor(num_gpus_per_worker)
else:
    num_gpus_per_worker = 0

seed = 42
setup_seed(seed)
num_test = 2
env = ddz.Game(seed)
env.set_agents(agents)
env.reset()


self_play_workers = [
                    self_play_test.SelfPlayTest.options(
                                            num_cpus=0,
                                            num_gpus=num_gpus_per_worker if config.train_on_gpu else 0,
                                        ).remote(env, config, configs, seed, agents)
                                        for _ in range(config.num_workers)
                    ]


result = [[],[],[]]
for i in range(config.num_workers):
    game_historys = ray.get(
                    self_play_workers[i].test.remote(
                                    num_test,
                                    0,
                                    0,
                                    False,
                                    "self",
                                    0,
                    )
                )
    for j in range(num_test):
        for i in range(3):
            result[i].append(
                sum(
                    reward
                    for k, reward in enumerate(game_historys[j][i].reward_history)
                )
            )

res_last = []
for i in range(3):
    res_last.append(numpy.mean(result[i]))

print(res_last)
