import torch
import torch.optim as optim
from torch.autograd import Variable
import random
import torch.nn as nn
from config import *

def train_dqn(gomoku, dqn, target_dqn, lr=0.001, batch_size=64, num_epochs=1000, self_play_rounds=1000, update_target_interval=50):
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        memory = []
        for round in range(self_play_rounds):
            gomoku.reset()
            while True:
                valid_moves = gomoku.get_valid_moves()
                if len(valid_moves) == 0:
                    break
                move = random.choice(valid_moves)
                winner = gomoku.make_move(move)
                if winner != 0:
                    break
                memory.append((gomoku.board.copy(), move, winner))

        for batch in range(len(memory) // batch_size):
            boards, moves, rewards = zip(*memory[batch * batch_size:(batch + 1) * batch_size])
            boards = torch.tensor(boards, dtype=torch.float32).unsqueeze(1)
            moves = torch.tensor(moves, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32).to('cuda')

            # logger.info(f"""{boards.shape}""")
            boards = boards.to('cuda')
            q_values = dqn(boards)
            target_q_values = target_dqn(boards).detach()

            q_values = q_values[moves[:, 0], moves[:, 1]]
            target_q_values = target_q_values.max(1)[0]

            loss = criterion(q_values, rewards + 0.99 * target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % update_target_interval == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        print(f"Epoch {epoch}, Loss: {loss.item()}")
