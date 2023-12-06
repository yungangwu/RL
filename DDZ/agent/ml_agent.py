import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Net).__init__(*args, **kwargs)

        # hand_cards_encode = nn.Linear()
        # desk_record_encode
        # 定义几个编码，手牌编码，打牌记录编码(LSTM)，上一个玩家的动作，及其他玩家手牌数量的信息，地主信息，炸弹数量，编码