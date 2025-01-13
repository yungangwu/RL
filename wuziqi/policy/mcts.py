import numpy as np
import copy
from config import *
from env import GomokuEnv

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)

def rollout_policy_fn(board: GomokuEnv):
    action_probs = np.random.rand(len(board.availables)) #生成一个长度为board.availables的随机数组，每个元素都在0，1之间
    return zip(board.availables, action_probs)

def policy_value_fn(board: GomokuEnv):
    action_probs = np.ones(len(board.availables)) / len(board.availables) # 所有的合法动作的概率都是一样的，没有倾向性
    return zip(board.availables, action_probs), 0


class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent: TreeNode = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)) # c_puct是一个控制计算价值方式的参数，一般越大，倾向于选择访问次数少的点

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct): # 用于计算每个节点的价值，ucb公式
        # 每个节点的ucb值计算，UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N)，Q代表节点的平均值，p代表先验概率，N_parent代表父节点的访问次数，N代表当前节点的访问次数，ucb公式中的被除数代表该节点的探索价值，值越大代表对探索过程的贡献越大
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS:
    def __init__(self, policy_value_fn, c_puct=5) -> None:
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct # c_puct决定探索的程度

    def _playout(self, state: GomokuEnv):
        node = self._root
        while (1):
            if node.is_leaf():
                break

            action, node = node.select(self._c_puct)
            state.step(action)

        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()

        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.current_player else -1.0
                )

        node.update_recursive(-leaf_value)

    def _playout_p(self, state: GomokuEnv): # 为了避免对state产生实际影响，这里的state必须是深拷贝的
        node = self._root
        while (1):
            if node.is_leaf():
                break

            action, node = node.select(self._c_puct) # 选择所有子节点中ucb值最大的那个节点代表的动作
            state.step(action)

        action_probs, _ = self._policy(state) # 到了叶子节点后用一个简单的策略，来计算出所有动作的概率
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)

        leaf_value = self._evaluate_rollout(state)
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, env, limit=1000): # 这一步就是原始mcts的模拟过程，目的是来得到当前棋面的价值，加入神经网络之后就直接预测这个价值，而不用实际的去模拟了
        player = env.current_player
        for i in range(limit):
            end, winner = env.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(env) # 用于模拟时，给未扩展的叶子节点生成一个随机的概率分布
            max_action = max(action_probs, key=itemgetter(1))[0]
            env.step(max_action)
        else:
            print("WARNING: rollout reached move limit")

        if winner == -1:
            return 0
        else:
            return 1 if winner == player else -1

    def get_move_probs(self, state: GomokuEnv, temp=1e-3):
        for n in range(n_playout):
            state_copy = copy.deepcopy(state) # 这里的state直接是整个环境env
            self._playout(state_copy)

        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits) # 将act_visits进行解压缩，分别存储在两个元组中
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10)) # 将动作的访问次数转化为概率分布，先将访问次数转化为对数，并用temp调节平滑，在使用softmax转化为概率分布

        return acts, act_probs

    def get_move(self, state):
        for n in range(n_playout):
            state_copy = copy.deepcopy(state)
            self._playout_p(state_copy)

        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0) # 主要是为纯MCTS用的

    def __str__(self):
        return "MCTS"

class MCTS_Pure:
    def __init__(self) -> None:
        self.mcts = MCTS(policy_value_fn, c_puct) # 默认的mcts中policy_value_fn，是一个所有动作的平均概率，对所有合法动作没有倾向性

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)

class MCTSPlayer(MCTS_Pure):
    def __init__(self, policy_value_function, is_selfplay=0):
        super(MCTS_Pure, self).__init__()
        self.mcts = MCTS(policy_value_function, c_puct)
        self._is_selfplay = is_selfplay

    def get_action(self, env, return_prob=0):
        sensible_moves = env.availables
        move_probs = np.zeros(board_width * board_width)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temperature)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(
                    acts,
                    p=noise_eps * probs + (1 - noise_eps) * np.random.dirichlet( # 对输出动作概率分布加噪声，使用加噪之后的概率选择动作
                        dirichlet_alpha * np.ones(len(probs))
                    )
                )
                self.mcts.update_with_move(move) # 根据当前的选择，更新蒙特卡洛树，将当前选择的动作作为根节点
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
