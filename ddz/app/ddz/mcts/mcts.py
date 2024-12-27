from ddz.game.utils import *
import numpy as np
import copy, time
import tensorflow as tf
from ddz.game.move import move_manager
from ddz.mcts.state import State
from ddz.model.network_defines import ACTION_DIM
# from graphviz import Digraph
from tensorflow.python.keras.engine import training
from collections import defaultdict

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

MAX_CHOICE = 20
class TreeNode(object):
    def __init__(self, mcts, parent, prior_p):
        self._mcts = mcts
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._c_puct_factor = 1.0
        self._state = None

    def set_state(self, state):
        self._state = state

    def get_state(self):
        return self._state

    def is_end(self):
        return self._state and self._state.end()

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        # this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._c_puct_factor * self._P * np.sqrt(self._parent._n_visits + 1) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

    def get_children(self):
        return self._children

    def get_visits(self):
        return self._n_visits

    def get_Q(self):
        return self._Q

    def get_P(self):
        return self._P

    def get_puct_factor(self):
        return self._c_puct_factor

    def __str__(self):
        texts = [
            "visits:{} | Q:{} | U:{} | P:{} | child:{} | leaf:{} | root:{}".format(
                self._n_visits, self._Q, self._u, self._P, len(self._children), self.is_leaf(), self.is_root()),
            "\nchild:"
        ]
        for k,_ in self._children.items():
            texts.append(str(k) + "|")
        return ''.join(texts)


class DecisionNode(TreeNode):
    def __init__(self, mcts, parent, prior_p):
        super(DecisionNode, self).__init__(mcts, parent, prior_p)


    def set_state(self, state):
        if state.end() and not state.is_winner(state.position):
            self._c_puct_factor = 10.0
        return super().set_state(state)

    def expand(self, action_priors):
        # assert not state.end()
        # print("=====================================")
        # state.print()
        state = self._state
        legal_actions_mask = state.get_legal_actions()
        not_legal_actions_mask = ~legal_actions_mask

        action_priors = action_priors + 1e-6
        action_priors[not_legal_actions_mask] = 0
        action_priors = action_priors / np.sum(action_priors)

        for action in range(1, ACTION_DIM):
            prob = action_priors[action]
            if prob > 0 and action not in self._children and legal_actions_mask[action]:
                state_copy = State(state.position)
                state_copy.copy_from_state(state)
                state_copy.do_move(action)
                # print("expand:", action)
                # state_copy.print()
                # print("========================================")
                p = 1.0 if state_copy.end() else prob
                node = ChanceNode(self._mcts, self, p)
                self._children[action] = node
                node.expand(state_copy)

    def select(self, c_puct):
        for k,v in self._children.items():
            v.do_expand()
        # for k,v in self._children.items():
        #     print(k)
        #     print(v)
        return super(DecisionNode, self).select(c_puct)

    def __str__(self):
        text = super(DecisionNode, self).__str__()
        texts = [
            "decision node\n",
            text,
        ]
        for k, v in self._children.items():
            texts.append('\n' + str(v))
        return "".join(texts)


class ChanceNode(TreeNode):
    def __init__(self, mcts, parent, prior_p):
        super(ChanceNode, self).__init__(mcts, parent, prior_p)
        self._expand = False

    def expand(self, state):
        self._state = state
        self._expand = False

    def do_expand(self):
        if not self._expand:
            state = self._state
            # print("do expand")
            # state.print()
            # print("========================================")
            # start = time.time()
            actions, probs = zip(*self._mcts.evaluate_infer(state))
            probs = np.array(probs)
            # print('time taken for eval is {} sec\n'.format(time.time()-start))
            limit_node_num = min(MAX_CHOICE, len(actions))
            action_indexs = np.argpartition(probs, -limit_node_num)[-limit_node_num:]
            probs = probs[action_indexs]
            # probs = softmax(1.0/2.0 * np.log(probs))
            probs = probs/np.sum(probs)

            for n, action_index in enumerate(action_indexs):
                action_seq = actions[action_index]
                if action_seq not in self._children and probs[n] > 0:
                    decision_node = DecisionNode(self._mcts, self, probs[n])
                    self._children[action_seq] = decision_node
                    state_decision = State(state.position)
                    state_decision.copy_from_state(state)
                    for action in action_seq:
                        if not action:
                            break
                        state_decision.do_move(action)
                    decision_node.set_state(state_decision)
            self._expand = True

    def __str__(self):
        text = super(ChanceNode, self).__str__()
        texts = [
            "chance node\n",
            text,
        ]
        # for k, v in self._children.items():
        #     texts.append('\n' + str(v))
        return "".join(texts)

class MCTS(object):

    def __init__(self, position, policy_net, infer_net, c_puct=5):
        self._root = DecisionNode(self, None, 1.0)
        self._position = position
        self._policy_net = policy_net
        self._infer_net = infer_net
        self._c_puct = c_puct

    def evaluate_policy(self, state):
        policy_input = state.get_policy_input()
        policy_input = np.expand_dims(policy_input, axis=0)
        action_porbs, value = self._policy_net.predict_on_batch(policy_input)
        action_porbs = action_porbs[0]
        value = value[0][0]
        return action_porbs, value

    def evaluate_infer(self, state):
        if state.end():
            return [[(0, 0), 1]]
        # print("==============================")
        # state.print()
        legal_actions_mask_one = state.get_legal_actions()
        action_states = [(0, state, legal_actions_mask_one)]

        for action_one in range(1, ACTION_DIM):
            if legal_actions_mask_one[action_one]:
                state_copy = State(state.position)
                state_copy.copy_from_state(state)
                state_copy.do_move(action_one)
                # print("=======================================")
                # state_copy.print()
                if state_copy.end():
                    action_states.append((action_one, None, None))
                else:
                    legal_actions_mask_two = state_copy.get_legal_actions()
                    action_states.append((action_one, state_copy, legal_actions_mask_two))

        ar = []
        for action_state in action_states:
            if action_state[1]:
                ar.append(action_state[1].get_infer_input())
        infer_inputs = np.stack(ar, axis=0)
        state_probs = self._infer_net.predict_on_batch(infer_inputs)

        index = 0
        not_legal_actions_mask_one = ~legal_actions_mask_one
        action_probs_one = state_probs[index] + 1e-6
        action_probs_one[not_legal_actions_mask_one] = 0
        action_probs_one = action_probs_one / np.sum(action_probs_one)
        index = index + 1
        state_index = index
        action_probs = []
        for action_one in range(1, ACTION_DIM):
            if legal_actions_mask_one[action_one]:
                prob_one = action_probs_one[action_one]
                assert action_states[index][0] == action_one
                state_copy = action_states[index][1]
                if not state_copy:
                    action_probs.append([(action_one, 0), prob_one])
                else:
                    action_probs_two = state_probs[state_index] + 1e-6
                    legal_actions_mask_two = action_states[index][2]
                    not_legal_actions_mask_two = ~legal_actions_mask_two
                    action_probs_two[not_legal_actions_mask_two] = 0
                    action_probs_two = action_probs_two / np.sum(action_probs_two)
                    state_index = state_index + 1
                    for action_two in range(1, ACTION_DIM):
                        prob_two = action_probs_two[action_two]
                        if legal_actions_mask_two[action_two]:
                            action_probs.append([(action_one, action_two), prob_one * prob_two])
                index = index + 1
        return action_probs

    def _playout(self, state):
        node = self._root
        node.set_state(state)
        # print("playput:")
        # state.print()
        # print("========================================")
        while(1):
            if node.is_leaf():
                break
            ## move to chance node
            action, chance_node = node.select(self._c_puct)
            chance_action, node = chance_node.select(self._c_puct)
            ## action with three step

        state = node.get_state()
        end = state.end()
        if not end:
            action_probs, leaf_value = self.evaluate_policy(state)
            node.expand(action_probs)
        else:
            leaf_value = (1.0 if state.is_winner(self._position) else -1.0)
        # print("update leaf value:", leaf_value)
        node.update_recursive(leaf_value)


    def get_action_probs(self, state, playout, temp=1e-3):
        # start = time.time()
        for n in range(playout):
            state_copy = State(self._position)
            state_copy.copy_from_state(state)
            self._playout(state_copy)
            # input("Press Enter to continue...")

        # print('time taken for p {} playput {} is {} sec\n'.format(self._position, n+1, time.time()-start))
        # self.plot(self._root, './graphs/mcts_graph_{}'.format(time.time()))
        # input("Press Enter to continue...")
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def match_env(self, env):
        state = self._root.get_state()
        if state:
            return state.match_env(env)
        return True

    def reset_root(self):
        self._root = DecisionNode(self, None, 1.0)

    def step(self, last_three_step):
        last_mine_move = last_three_step[-1]
        found_root = False
        if last_mine_move in self._root._children:
            root = self._root._children[last_mine_move]
            last_infer_move = (last_three_step[-2], last_three_step[-3])
            if last_infer_move in root._children:
                self._root = root._children[last_infer_move]
                self._root._parent = None
                found_root = True
                if DEBUG:
                    print("step ok:", last_three_step[-1::-1])
        if not found_root:
            self._root = DecisionNode(self, None, 1.0)
            if DEBUG:
                print("step fail")
        return found_root

    # def plot(self, node, save_path='./graphs/mcts_graph'):
    #     MCTS_GRAPHA = Digraph(comment='mcts_graph')
    #     counter = defaultdict(int)

    #     root = str(None)
    #     counter[root] += 1
    #     MCTS_GRAPHA = Digraph(str(root) + str(counter[root]), 'ROOT')

    #     def plot_mcts(node, root=None):
    #         if not root:
    #             root = str(None)

    #         if not node.is_leaf():
    #             children = node.get_children()
    #             max_child = max(children, key=lambda child: children[child].get_visits())
    #             for child, child_node in children.items():
    #                 counter[child] += 1
    #                 P = child_node.get_P()
    #                 if P >= 1e-2:
    #                     child_index = str(child) + '_' + str(counter[child])
    #                     if isinstance(child, int):
    #                         move = move_manager.get_move_by_id(child)
    #                         act_str = cards_value_to_str(move.get_action_cards())
    #                     else:
    #                         act_str = ''
    #                         for action in child:
    #                             if action > 0:
    #                                 if action == 309:
    #                                     act_str = act_str + 'pass|'
    #                                 else:
    #                                     move = move_manager.get_move_by_id(action)
    #                                     act_str = act_str + cards_value_to_str(move.get_action_cards()) + '|'
    #                     label = ('act: {}'.format(act_str) + '\nvisit: {}'.format(str(child_node.get_visits())) +
    #                                 '\nvalue:{:.2f}'.format(child_node.get_Q()) +
    #                                 '\np:{:.6f}'.format(child_node.get_P()) +
    #                                 '\nv:{:.6f}'.format(child_node.get_value(self._c_puct)) +
    #                                 '\nf:{:.2f}'.format(child_node.get_puct_factor()))
    #                     color = ''
    #                     if child_node.is_end():
    #                         color = 'red'
    #                     # 修改label，打印需要显示的结果
    #                     if isinstance(child_node, ChanceNode):
    #                         shape='box'
    #                         if not color:
    #                             MCTS_GRAPHA.node(child_index, label, shape=shape)
    #                         else:
    #                             MCTS_GRAPHA.node(child_index, label, shape=shape, color=color)
    #                     else:
    #                         if not color:
    #                             MCTS_GRAPHA.node(child_index, label)
    #                         else:
    #                             MCTS_GRAPHA.node(child_index, label, color=color)

    #                     if child == max_child and (len(children) > 1):
    #                         MCTS_GRAPHA.edge(root, child_index, color='blue')
    #                     else:
    #                         MCTS_GRAPHA.edge(root, child_index)
    #                     plot_mcts(child_node, child_index)

    #     plot_mcts(node)
    #     # current_time = time.clock()
    #     MCTS_GRAPHA.render(save_path, view=True)
    #     # whether pop out a picture
    #     return MCTS_GRAPHA


    def __str__(self):
        return "MCTS"