from ddz.mcts.state import State
from ddz.environment.env import Env
from ddz.agent.agent_mcts import AgentMCTS
from ddz.agent.agent_net import AgentNet
from ddz.agent.agent_random import AgentRandom
from ddz.train.game_record import GameRecord
from ddz.policy.policy_registry import *
from ddz.game.utils import *
from ddz.predict.predict_winrate import PredictWinrate
import random, time
import numpy as np

def battle(agents, episode, seed=None, predictor=None):
    env = Env()
    win_count = [0 for _ in range(NUM_AGENT)]
    for n, agent in enumerate(agents):
        env.set_agent(agent, n)
    for i in range(episode):
        if seed != None:
            # print("seed:", seed[i])
            env.reset(seed[i])
        else:
            env.reset()
        if predictor:
            cards = list(map(lambda x:encode_card(x), agents[0].handcards))
            win = predictor.predict_winrate(cards)
            print("predict win of {}:{}".format(n, win > 0))
        done = False
        position = 0
        # start = time.time()
        while not done:
            if DEBUG:
                print("********************************************")
            agent = env.get_agent(position)
            cards = agent.get_action()
            done = env.step(position, cards)
            position = (position + 1) % 3
        # print("episode {}:winner is:{}, time:{}".format(i + 1, env.winner, time.time() - start))
        for n in env.winner:
            win_count[n]+=1

        if (i+1) % 100 == 0:
            print("result of epoch:{}".format(i+1))
            for k in range(NUM_AGENT):
                print("win count of position [{}] is:{} winrate:{:.2f}".format(k, win_count[k], float(win_count[k])/(i+1)))
    
    return win_count


def arena_lord(episode, seed, lords, peasant, path):
    if seed:
        random.seed(seed)  
    seeds = [random.randint(0, 100000) for _ in range(episode)]
    best_lord = -1
    max_win = 0
    for lord in lords:
        lord_model_path = os.path.join(path, 'v_{}'.format(lord))
        peasant_model_path = os.path.join(path, 'v_{}'.format(peasant))
 
        agents = [AgentNet(model_path=lord_model_path), 
            AgentNet(model_path=peasant_model_path), AgentNet(model_path=peasant_model_path)]
        print("{} vs {}".format(lord_model_path, peasant_model_path))
        print("==========================================================")
        wincount = battle(agents, episode, seeds)
        if wincount[0] > max_win:
            best_lord = lord
            max_win = wincount[0]
    return best_lord

def arena_peasant(episode, seed, peasants, lord, path):
    if seed:
        random.seed(seed)  
    seeds = [random.randint(0, 100000) for _ in range(episode)]
    best_peasant = -1
    max_win = 0
    for peasant in peasants:
        lord_model_path = os.path.join(path, 'v_{}'.format(lord))
        peasant_model_path = os.path.join(path, 'v_{}'.format(peasant))
        agents = [AgentNet(model_path=lord_model_path), 
            AgentNet(model_path=peasant_model_path), AgentNet(model_path=peasant_model_path)]
        print("==========================================================")
        print("{} vs {}".format(lord_model_path, peasant_model_path))
        wincount = battle(agents, episode, seeds)
        if wincount[1] > max_win:
            best_peasant = peasant
            max_win = wincount[1]
    return best_peasant