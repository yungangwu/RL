from util.maze_env import Maze
from Sarsa_RL_brain import SaraTable

def update():
    for episode in range(100):
        observation = env.reset()

        action = RL.choose_action(str(observation))

        # sarsa_lambda的步骤
        RL.eligibility_trace *= 0

        while True:
            env.render()

            observation_, reward, done = env.step(action)

            action_ = RL.choose_action(str(observation_))

            RL.learn(str(observation), action, reward, str(observation_), action_)

            observation = observation_

            action = action_

            if done:
                break
    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = SaraTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()