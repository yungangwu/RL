from utils import *
from model.networks import *
from policy.minor_cards_rules import *
from move import move_manager, get_legal_actions


class State(object):
    def __init__(self, position):
        self.position = position
        self.winner = None

    def copy_from_env(self, env):
        agent = env.get_agent(self.position)
        self.env = env
        self.current_round = env.round
        self.handcards = agent.handcards.copy()
        self.actions = [[] for _ in range(NUM_AGENT)]
        remain_cards = env.get_remain_cards()
        self.remain_cards = [x for x in remain_cards if x not in self.handcards]
        self.playedout_cards = env.played_cards.copy()
        for i in range(NUM_AGENT):
            agent = env.get_agent(i)
            self.actions[i] = agent.get_history_actions()
        self._check_winner()
        # assert (len(self.handcards) + len(self.remain_cards) + len(self.playedout_cards)) == 54 

    def match_env(self, env):
        for i in range(NUM_AGENT):
            state_actions = self.actions[i].copy()
            agent = env.get_agent(i)
            env_actions = agent.get_history_actions()
            for state_action, env_action in zip(state_actions, env_actions):
                state_action = get_card_values(state_action)
                env_action   = get_card_values(env_action)
                if state_action != env_action:
                    return False
        return True

    def copy_from_state(self, state):
        self.env = state.env
        self.current_round = state.current_round
        self.position = state.position
        self.handcards = state.handcards.copy()
        self.remain_cards = state.remain_cards.copy()
        self.playedout_cards = state.playedout_cards.copy()
        self.actions = [state.actions[i].copy() for i in range(NUM_AGENT)]
        self._check_winner()


    def handout(self, cards):
        position = self.current_round
        if position == self.position:
            pickout_cards_src = self.handcards
        else:
            pickout_cards_src = self.remain_cards
        pickout_cards = []
        cards_copy = cards.copy()
        while cards_copy:
            card_value = cards_copy.pop()
            for card in pickout_cards_src:
                if card[1] == card_value:
                    pickout_cards.append(card)
                    pickout_cards_src.remove(card)
                    break
        if len(cards) != len(pickout_cards):
            raise RuntimeError("can not handout cards:{}".format(cards_value_to_str(cards)))
        pickout_cards.sort(key=lambda k: k[1])
        actions = self.actions[position]
        actions.append(pickout_cards)
        self.playedout_cards.extend(pickout_cards)

    def do_move(self, action):
        action_type, action_index = decode_action(action)
        move = move_manager.get_move_by_id(action)
        cards = move.get_action_cards()
        minor_card_count = get_request_minor_card_count(
            action_type, action_index)
        if minor_card_count:
            if self.current_round == self.position:
                agent = self.env.get_agent(self.position)
                for card_count in minor_card_count:
                    minor_cards = agent.get_minor_cards(self, card_count, cards)
                    cards.extend(minor_cards)
            else:
                for card_count in minor_card_count:
                    minor_card = get_minor_cards_by_rules(self.remain_cards, card_count, cards)
                    cards.extend([minor_card for _ in range(card_count)])
        self.handout(cards)
        self._check_winner()
        self.current_round = (self.current_round + 1) % 3

    def _check_winner(self):
        total_cards_count = [20, 17, 17]
        for i in range(NUM_AGENT):
            handout_count = 0
            for action in self.actions[i]:
                handout_count = handout_count + len(action)
            if handout_count >= total_cards_count[i]:
                if i == 0:
                    self.winner = [0]
                else:
                    self.winner = [1, 2]
                return

    def end(self):
        return self.winner != None
    
    def clamp_position(self, position):
        position = (position + NUM_AGENT) % NUM_AGENT
        return position

    def get_remain_card_num(self):
        remain_card_nums = [20, 17, 17]
        for i in range(NUM_AGENT):
            count = 0
            for action in self.actions[i]:
                count = count + len(action)
            remain_card_nums[i] = remain_card_nums[i] - count
        return remain_card_nums

    def encode_remain_handcards_num(self):
        onehot_code = np.zeros(60, dtype=np.float32)
        remain_card_nums = self.get_remain_card_num()
        for i in range(NUM_AGENT):
            subvec = np.zeros(20, dtype=np.float32)
            subvec[:remain_card_nums[i]] = 1
            onehot_code[i*20:(i+1)*20] = subvec
        return onehot_code

    def get_legal_actions(self):
        current_round = self.current_round
        position = current_round
        prev_action = None
        for i in range(2):
            prev_position = self.clamp_position(position-1-i)
            if self.actions[prev_position]:
                prev_action = self.actions[prev_position][-1]
                if prev_action:
                    break
        if prev_action:
            prev_action_id = move_manager.get_move_by_cards(get_card_values(prev_action))
        else:
            prev_action_id = None
        test_cards = self.handcards if current_round == self.position else self.remain_cards
        legal_actions = get_legal_actions(test_cards, prev_action_id)
        mask = np.zeros(310, dtype=np.bool)
        for action in legal_actions:
            mask[action] = True
        return mask

    def get_last_round_cards(self, position):
        ar = []
        for i in range(NUM_AGENT):
            current_position = self.clamp_position(position-1-i)
            actions = self.actions[current_position]
            if actions:
                ar.append(cards_to_onehot_60(actions[-1]))
                # print("last position {} a:{}".format(current_position, cards_to_str(actions[-1])))
            else:
                ar.append(card_values_to_onehot_60([]))
                # print("last position {} a:".format(current_position))
        return ar

    def get_agent_actions(self, position):
        actions = self.actions[position]
        action_cards = []
        for action in actions:
            action_cards.extend(action)
        return action_cards

    def get_agent_memories(self, memory_size):
        ar = []
        for k in range(memory_size):
            for i in range(NUM_AGENT):
                current_position = self.clamp_position(self.position-1-i)
                actions = self.actions[current_position]
                if actions and (k+1) <= len(actions):
                    ar.append(cards_to_onehot_60(actions[-(k+1)]))
                    # print("agent {} action: {}".format(current_position, cards_to_str(actions[-(k+1)])))
                else:
                    ar.append(card_values_to_onehot_60([]))
                    # print("agent {} action: none".format(current_position))
        return ar

    def get_policy_input(self):
        ar = []
        last_round_cards = self.get_last_round_cards(self.position)
        ar.append(cards_to_onehot_60(self.handcards))
        ar.append(cards_to_onehot_60(self.remain_cards))
        ar.extend(last_round_cards)
        ar.append(cards_to_onehot_60(self.playedout_cards))
        remain_handcards_num = self.encode_remain_handcards_num()
        ar.append(remain_handcards_num)
        s = np.stack(ar, axis=-1)
        return s

    def get_infer_input(self):
        ar = []
        # print("=====================================================")
        # print("current_round:", self.current_round)
        ar.append(cards_to_onehot_60(self.remain_cards))
        # print("remain cards:{}".format(cards_to_str(self.remain_cards)))
        infer_position = self.current_round
        actions = self.actions[infer_position]
        action_cards = []
        for action in actions:
            action_cards.extend(action)
        ar.append(cards_to_onehot_60(action_cards))
        # print("pos {} actions:{}".format(infer_position, cards_to_str(action_cards)))
        # print("agent {} actions:{}".format(infer_position, cards_to_str(action_cards)) )

        for i in range(NUM_AGENT):
            if i != infer_position and i != self.position:
                other_position = i
                break

        actions = self.actions[other_position]
        action_cards = []
        for action in actions:
            action_cards.extend(action)
        ar.append(cards_to_onehot_60(action_cards))
        # print("pos {} actions:{}".format(other_position, cards_to_str(action_cards)))

        last_round_cards = self.get_last_round_cards(infer_position)
        ar.extend(last_round_cards)

        remain_handcards_num = self.encode_remain_handcards_num()
        ar.append(remain_handcards_num)
        
        # print("remain_handcards_num", remain_handcards_num)
        # print("agent {} actions:{}".format(other_position, cards_to_str(action_cards)) )
        s = np.stack(ar, axis=-1)
        return s

    def is_winner(self, position):
        return position in self.winner

    def print(self):
        print("agent [{}]'s state:".format(self.position))
        print("round:{}".format(self.current_round))
        print("handcards:{}".format(cards_to_str(self.handcards)))
        print("remain_cards:{}".format(cards_to_str(self.remain_cards)))
        print("playedout_cards:{}".format(cards_to_str(self.playedout_cards)))
        if self.winner:
            print("winner:{}".format(self.winner))
        for i in range(NUM_AGENT):
            prev_position = self.clamp_position(self.current_round-1-i)
            if self.actions[prev_position]:
                action = self.actions[prev_position][-1]
                print("last action of {}:{}".format(prev_position, cards_to_str(action) if action else "pass"))
            else:
                print("last action of {}: None".format(prev_position))

        print("legal actions:{}", np.where(self.get_legal_actions()))
        print("remain cards:", self.get_remain_card_num())
            
