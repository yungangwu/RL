import requests
from ddz.environment.env import Env
from ddz.agent.agent_remote import AgentRemote
from ddz.game.utils import *

def predict_winrate(endpoint, handcards):
    cards = handcards.copy()
    cards = list(map(lambda x:encode_card(x), cards))
    data = {
        'cards': cards,
    }
    response = requests.post(url = endpoint + '/ai/ddz/predict_dizhu', json = data)
    if response.status_code == 200:
        return response.json()['dizhu']
    else:
        return 0

def test_remote_server(endpoint, agent_level, episode, seed=None):
    env = Env()
    agents = [AgentRemote(endpoint) for _ in range(NUM_AGENT)]
    for n, agent in enumerate(agents):
        env.set_agent(agent, n)
    for i in range(episode):
        if seed != None:
            env.reset(seed[i])
        else:
            env.reset()
        data = {
            'desk_id':i+1,
        }
        agent_config = []
        for n in range(NUM_AGENT):
            agents[n].set_desk_id(i+1)
            cards = agents[n].initial_cards.copy()
            cards = list(map(lambda x:encode_card(x), cards))
            agent_data = {
                'position':n,
                'level':agent_level[n],
                'cards': cards,
            }
            agent_config.append(agent_data)
            winrate = predict_winrate(endpoint, agents[n].initial_cards)
            print("winrate [{}]:{}".format(n, winrate))
        data['agents'] = agent_config
        response = requests.post(url = endpoint + '/ai/ddz/create_desk', json = data)
        if response.status_code == 200:
            done = False
            position = 0
            while not done:
                if DEBUG:
                    print("********************************************")
                agent = env.get_agent(position)
                cards = agent.get_action()
                done = env.step(position, cards)
                position = (position + 1) % 3

            print("episode {}:winner is:{}".format(i + 1, env.winner))
            data = {
                'desk_id':i+1,
            }
            response = requests.post(url = endpoint + '/ai/ddz/destroy_desk', json = data)
            print("destroy desk {}:{}".format(i+1, response.status_code))
        else:
            print('get error response:', response)