from ddz.agent.agent import Agent
from ddz.game.utils import *
import requests

class AgentRemote(Agent):
    def __init__(self, remote_url):
        super(AgentRemote, self).__init__()
        self._url = remote_url

    def set_desk_id(self, desk_id):
        self._desk_id = desk_id

    def get_action(self):
        data = {
            'desk_id':self._desk_id,
            'position':self.position
        }
        response = requests.post(url = self._url +  '/ai/ddz/action', json = data)
        if response.status_code == 200:
            cards = response.json()['aiPlayCards']
            if cards:
                return list(map(lambda x:decode_card(x), cards))
        return []

    def handout_cards(self, cards):
        handout_cards = super(AgentRemote, self).handout_cards(cards)
        encode_cards = []
        if cards:
            encode_cards = list(map(lambda x:encode_card(x), cards))
        data = {
            'desk_id': self._desk_id,
            'position': self.position,
            'cards': encode_cards,
        }
        requests.post(url = self._url +  '/ai/ddz/step', json = data)
        return handout_cards