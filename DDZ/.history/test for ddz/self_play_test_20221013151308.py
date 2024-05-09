import math
from telnetlib import GA
import time

import numpy
import ray
import torch

import models

seed = 42
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     numpy.random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(seed)

@ray.remote
class SelfPlayTest:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, Game, config, configs, seed, agents):
        self.config = config
        self.configs = configs
        self.game = Game
        self.agents = agents
        
        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        # Creating a game_historys list, include num_players' game_history.
        game_historys = []
        for i in range(self.config.num_players):
            game_history = GameHistory()
            game_historys.append(game_history)
        observation = self.game.reset()
        # game_historys[0].action_history.append(0)
        game_historys[0].observation_history.append(observation)
        # game_historys[0].reward_history.append(0)
        game_historys[0].to_play_history.append(self.game.to_play())
            
        
        done = False
        count = 0

        if render:
            self.game.render()

        player_id = self.game.to_play()
        rewards = self.config.reward
        flag = True

        with torch.no_grad():
            while flag:
                # print('第{}次-------'.format(count // 3))
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.agents[player_id].observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.agents[player_id].observation_shape} but got {numpy.array(observation).shape}."
                stacked_observations = game_historys[player_id].get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )

                # print('stacked_observations',stacked_observations)
                # Choose the action
                if opponent == "self" or muzero_player == self.game.to_play():
                    # print('做出动作选择')
                    root, mcts_info = MCTS(self.config).run(
                        self.agents[player_id],
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.to_play(),
                        False,
                        count,
                    )
                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_historys[player_id].action_history) < temperature_threshold
                        else 0,
                    )

                    if render:
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(
                            f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                        )
                else:
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations
                    )

                # print('action-----',action)
                observation, reward, done = self.game.step(action, player_id)

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_historys[player_id].store_search_statistics(root, self.config.action_space)
                

                # Next batch
                if not done:
                    reward_p = reward[player_id] - rewards[player_id]
                    game_historys[player_id].action_history.append(action)
                    game_historys[player_id].reward_history.append(reward_p)

                    # The next player's game_history
                    player_id = self.game.to_play()
                    game_historys[player_id].to_play_history.append(player_id)
                    game_historys[player_id].observation_history.append(observation)
                else:
                    game_historys[player_id].action_history.append(action)
                    print(reward)
                    for i in range(self.config.num_players):
                        game_historys[i].reward_history.append(reward[i])
                    player_id = self.game.to_play()
                    game_historys[player_id].to_play_history.append(player_id)
                    game_historys[player_id].observation_history.append(observation)
                
                for i in range(self.config.num_players):
                    if not done:
                        flag = True
                    else:
                        flag = False
                
                count = count + 1
                

        return game_historys

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        player_id = self.game.to_play()
        if opponent == "human":
            root, mcts_info = MCTS(self.config).run(
                self.agents[player_id],
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                False,
                self.game,
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        # print('visit_counts', visit_counts, 'actions', actions)
        action = actions[numpy.argmax(visit_counts)]
        # if temperature == 0:
        #     action = actions[numpy.argmax(visit_counts)]
        # elif temperature == float("inf"):
        #     action = numpy.random.choice(actions)
        # else:
        #     # See paper appendix Data Generation
        #     visit_count_distribution = visit_counts ** (1 / temperature)
        #     visit_count_distribution = visit_count_distribution / sum(
        #         visit_count_distribution
        #     )
        #     action = numpy.random.choice(actions, p=visit_count_distribution)

        return action

def tensor_to_img(tensor,count):
    import matplotlib.pyplot as plt
    tensor = torch.squeeze(tensor,0).tolist()
    x = [i for i in range(len(tensor))]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x=x, height=tensor)
    ax.set_title("Simple Bar Plot", fontsize=15)
    plt.savefig('./act_img1/action_pro{}.jpg'.format(count))
    plt.show()
    # return tensor
    
def top_k_act(tensor):
    k_act_pro = torch.topk(tensor, 5)
    k_pol, k_act= k_act_pro.values, torch.squeeze(k_act_pro.indices).tolist()
    print(k_act)


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        count,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if legal_actions:
            legal_actions.sort()
        # print('legal_actions',legal_actions)
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            (
                root_predicted_value,
                reward,
                policy_logits,
                hidden_state,
            ) = model.initial_inference(observation)
            root_predicted_value = models.support_to_scalar(
                root_predicted_value, self.config.support_size
            ).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            # if count == 0:
            #     print('111111111111')
            #     top_k_act(policy_logits)
                # tensor_to_img(policy_logits, count)
                # print('policy_logits',tensor_pol.tolist())
            # print('legal_actions',legal_actions)
            # print('policy_logits',policy_logits.tolist())
            root.expand_for_org(
                legal_actions,
                to_play,
                reward,
                policy_logits,
                hidden_state,
            )

        # if add_exploration_noise:
        #     root.add_exploration_noise(
        #         dirichlet_alpha=self.config.root_dirichlet_alpha,
        #         exploration_fraction=self.config.root_exploration_fraction,
        #     )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            # print('重新开始模拟')
            while node.expanded(): #判断是否为叶子节点
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats) #选择ucb最大的动作，并将返回该动作的node子节点
                search_path.append(node) #搜索路径添加该子节点
                # print('action-s----',action)

                # Players play turn by turn
                # if virtual_to_play + 1 < len(self.config.players):
                #     virtual_to_play = self.config.players[virtual_to_play + 1]
                # else:
                #     virtual_to_play = self.config.players[0]

            # print('current_tree_depth',current_tree_depth)

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            # if count == 0:
            #     tensor_to_img(policy_logits, j)
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
 
            # print('policy_logits---------',policy_logits)
            # Select K candidate actions, to reduce the complex of the action space
            k_candidates = torch.topk(policy_logits, self.config.k_sample)
            k_policy_logits, k_actions = k_candidates.values, torch.squeeze(k_candidates.indices).tolist()

            # print('k_policy_logits-----',k_policy_logits,'k_actions',k_actions)
            node.expand( #当前节点进行扩展，下一层，挂载的属性有当前的奖励，
                k_actions,
                virtual_to_play,
                reward,
                k_policy_logits,
                hidden_state,
            )


            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        # print('max_ucb',max_ucb)
        actions = []
        for action, child in node.children.items():
            if self.ucb_score(node, child, min_max_stats) == max_ucb:
                actions.append(action)
        
        # print(actions)
        # numpy.random.seed(42)
        action = numpy.random.choice(actions)
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * node.value())

                value = node.reward + self.config.discount * value

        elif len(self.config.players) > 1:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")





class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    # def expand(self, actions, to_play, reward, policy_logits, hidden_state):
    #     """
    #     We expand a node using the value, reward and policy prediction obtained from the
    #     neural network.
    #     """
    #     self.to_play = to_play
    #     self.reward = reward
    #     self.hidden_state = hidden_state

    #     policy_values = torch.softmax(
    #         torch.tensor([policy_logits[0][a] for a in actions]), dim=0
    #     ).tolist()
    #     policy = {a: policy_values[i] for i, a in enumerate(actions)}
    #     for action, p in policy.items():
    #         self.children[action] = Node(p)

    def expand_for_org(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        # print('actions',actions,'policy_logits',policy_logits)
        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        # print('policy',policy)
        for action, p in policy.items():
            self.children[action] = Node(p)

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        # print('2222222222')
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in range(len(actions))]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        # print('policy',policy)
        for action, p in policy.items(): #所有子节点可选动作的概率值
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = numpy.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            numpy.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = numpy.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
