from ddz.game.utils import *
from ddz.agent.agent_human import AgentHuman
from ddz.agent.agent_mcts import AgentMCTS
from ddz.agent.agent_net import AgentNet
from ddz.environment.env_predict import EnvPredict
from ddz.predict.agent_level_config import agent_level_config

class Predictor(object):
    def __init__(self):
        super().__init__()
        
    def _get_agent(self, config):
        agent_type = config['type']
        model_path = config['model']
        if agent_type == 'mcts':
            return AgentMCTS(model_path, 3, 400, False)
        elif agent_type == 'net':
            temperature = config['temperature'] if 'temperature' in config else 0
            return AgentNet(model_path=model_path, temperature=temperature)
        else:
            raise RuntimeError("invalid agent type:{}".format(agent_type))

    def start_game(self, config):
        self._config = config.copy()
        self._agents = []
        initial_cards = []
        for i in range(NUM_AGENT):
            if i in config:
                level = config[i]['level']
                cards = config[i]['cards']
                cards = list(map(lambda x: decode_card(x), cards))
                initial_cards.append(cards)
                agent = self._get_agent(agent_level_config[level])
                self._agents.append(agent)
            else:
                self._agents.append(AgentHuman())
                initial_cards.append([])
        env = EnvPredict()
        for n, agent in enumerate(self._agents):
            env.set_agent(agent, n)
        env.reset(initial_cards)
        self._env = env
        self._done = False
        
    def env(self):
        return self._env

    def end(self):
        return self._done

    def get_agent(self, position):
        return self._agents[position]

    def step(self, position, cards):
        if not self._done:
            self._done = self._env.step(position, cards)
            if self._done:
                print("winner is:{}".format(self._env.winner))

    def _is_human(self, position):
        return position not in self._config

    def get_action(self, position):
        if self._env.round == position:
            if not self._is_human(position):
                agent = self._agents[position]
                handout_cards = agent.get_action()
                return handout_cards
            else:
                raise RuntimeError("can not get human agent action {}".format(position))
        else:
            raise RuntimeError("current {} is not your round {}".format(self._env.round, position))


class PredictorManager(object):
    def __init__(self):
        self._desks = {}

    def create_desk(self, desk_id, config):
        if desk_id not in self._desks:
            predictor = Predictor()
            predictor.start_game(config)
            self._desks[desk_id] = predictor
        else:
            raise RuntimeError("desk id {} duplicated.".format(desk_id))
            
    def step(self, desk_id, position, cards):
        if desk_id in self._desks:
            predictor = self._desks[desk_id]
            if cards:
                cards = list(map(lambda x: decode_card(x), cards))
                predictor.step(position, cards)
            else:
                predictor.step(position, [])
        else:
            raise RuntimeError("desk id {} is not found in step.".format(desk_id))

    def get_action(self, desk_id, position):
        if desk_id in self._desks:
            predictor = self._desks[desk_id]
            cards = predictor.get_action(position)
            cards = list(map(lambda x:encode_card(x), cards))
            return cards
        else:
            raise RuntimeError("desk id {} is not found in get_action.".format(desk_id))

    def destroy_desk(self, desk_id):
        if desk_id in self._desks:
            self._desks.pop(desk_id)
        else:
            raise RuntimeError("desk id {} is not found in destroy_desk.".format(desk_id))