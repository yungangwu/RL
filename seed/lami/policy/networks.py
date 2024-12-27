import torch
from torch import nn

class MCQNet(nn.Module):
    def __init__(self, input_dim=0):
        super().__init__()
        dense_layers = [input_dim, 512, 512, 512, 512, 512, 512, 1]
        layers = []
        for i in range(len(dense_layers) - 1):
            layers.append(nn.Linear(dense_layers[i], dense_layers[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, s):
        x = s
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if x != len(self.layers) - 1:
                x = torch.relu(x)
        return x

class CardsFeatureExtraction(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        liner_input_dim = 1
        for dim in input_dim:
            liner_input_dim *= dim
        self.liner = nn.Linear(liner_input_dim, output_dim)

    def forward(self, input):
        x = torch.reshape(input, (input.shape[0], -1))
        y = self.liner(x)
        y = torch.relu(y)
        return y

class AttentionNet(nn.Module):
    def __init__(self, handcards_dim, action_dim, feature_dim, atten_head, dense_layers, output_dim):
        super().__init__()
        self.handcards_fe = CardsFeatureExtraction(handcards_dim, feature_dim)
        self.action_fe = CardsFeatureExtraction(action_dim, feature_dim)
        self.attn = nn.MultiheadAttention(feature_dim, atten_head, batch_first=True)
        layers = []
        for i in range(len(dense_layers) - 1):
            layers.append(nn.Linear(dense_layers[i], dense_layers[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(dense_layers[-1], output_dim)
        self.flat_keys = ['handcards_num', 'gui_num', 'played_gui_num', 'played_A_num', 'is_free_play', 'is_lose']

    def forward(self, inputs, action):
        # 获取手中的牌
        handcards = inputs['handcards']
        # 对手中的牌进行特征提取
        handcards = self.handcards_fe(handcards)

        # 获取剩余的牌
        remain_cards = self.handcards_fe(inputs['remain_cards'])

        # 获取桌面上的牌
        desk_cards = inputs['desk_cards']
        # 获取桌面上的牌的维度信息
        desk_shape = desk_cards.shape
        # 将桌面上的牌展平为一维
        desk_cards = desk_cards.reshape((desk_shape[0] * desk_shape[1], *desk_shape[2:]))
        # 对桌面上的牌进行特征提取
        desk_cards = self.action_fe(desk_cards)
        # 将桌面上的牌恢复为原来的形状
        desk_cards = desk_cards.reshape((*desk_shape[0:2], *desk_cards.shape[1:]))

        # 初始化特征列表
        flat_features = []
        # 遍历需要展平的特征键
        for key in self.flat_keys:
            # 获取特征
            features = inputs[key]
            # 将特征展平为一维
            features = features.reshape((features.shape[0], -1))
            # 将展平后的特征添加到列表中
            flat_features.append(features)

        # 将特征列表拼接为一个张量
        flat_features = torch.hstack(flat_features)
        # 对动作进行特征提取
        action_features = self.action_fe(action)
        # 将手中的牌、剩余的牌、展平的特征和动作的特征拼接在一起
        features = torch.cat((handcards, remain_cards, flat_features, action_features), dim=-1)

        # 将动作的特征增加一个维度
        action_features = action_features.reshape((action_features.shape[0], 1, *action_features.shape[1:]))
        # 计算注意力输出和注意力权重
        attn_output, attn_output_weights = self.attn(action_features, desk_cards, desk_cards)
        # 将注意力输出展平为一维
        attn_output = attn_output.reshape((attn_output.shape[0], *attn_output.shape[2:]))

        # 将注意力输出和之前的特征拼接在一起
        features = torch.cat((attn_output, features), dim=-1)
        x = features
        # 遍历全连接层
        for linear in self.layers:
            # 对特征进行线性变换
            x = linear(x)
            # 对线性变换后的特征进行ReLU激活
            x = torch.relu(x)
        # 对特征进行最后的线性变换
        y = self.linear(x)
        return y

class LSTMAttentionNet(AttentionNet):
    def __init__(self, handcards_dim, action_dim, feature_dim, atten_head, dense_layers, output_dim, lstm_hidden_dim):
        super().__init__(handcards_dim, action_dim, feature_dim, atten_head, dense_layers, output_dim)
        self.lstm = nn.LSTM(feature_dim, lstm_hidden_dim, batch_first=True)

    def encode_seq(self, seq_feature):
        seq_shape = seq_feature.shape
        seq_feature = seq_feature.reshape((seq_shape[0] * seq_shape[1], *seq_shape[2:]))
        seq_feature = self.action_fe(seq_feature)
        seq_feature = seq_feature.reshape((*seq_shape[0:2], *seq_feature.shape[1:]))
        return seq_feature

    def forward(self, inputs, action):
        handcards = inputs['handcards']
        handcards = self.handcards_fe(handcards)

        remain_cards = self.handcards_fe(inputs['remain_cards'])

        desk_cards = inputs['desk_cards']
        desk_shape = desk_cards.shape
        desk_cards = desk_cards.reshape((desk_shape[0] * desk_shape[1], *desk_shape[2:]))
        desk_cards = self.action_fe(desk_cards)
        desk_cards = desk_cards.reshape((*desk_shape[0:2], *desk_cards.shape[1:]))

        flat_features = []
        for key in self.flat_keys:
            features = inputs[key]
            features = features.reshape((features.shape[0], -1))
            flat_features.append(features)

        flat_features = torch.hstack(flat_features)
        action_features = self.action_fe(action)
        features = torch.cat((handcards, remain_cards, flat_features, action_features), dim=-1)

        action_features = action_features.reshape((action_features.shape[0], 1, *action_features.shape[1:]))
        attn_output, attn_output_weights = self.attn(action_features, desk_cards, desk_cards)
        attn_output = attn_output.reshape((attn_output.shape[0], *attn_output.shape[2:]))

        history_feature = self.encode_seq(inputs['history_actions'])
        history_feature, (h_n, _) = self.lstm(history_feature)
        history_feature = history_feature[:, -1, :]

        features = torch.cat((attn_output, features, history_feature), dim=-1)
        x = features
        for linear in self.layers:
            x = linear(x)
            x = torch.relu(x)
        y = self.linear(x)
        return y

class CNNFeatureExtraction(nn.Module):
    def __init__(self, input_dim, output_channel) -> None:
        super().__init__()
        in_channel = input_dim[-1]
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=output_channel, kernel_size=(4, 3))

    def forward(self, input):
        input_shape = input.shape
        x = torch.reshape(input, (input_shape[0], input_shape[3], input_shape[1], input_shape[2]))
        y = self.conv(x)
        y = torch.relu(y)
        y = torch.reshape(y, (y.shape[0], -1))
        return y

class TransformerNet(nn.Module):
    def __init__(self, handcards_dim, action_dim, feature_dim, atten_head, dense_layers, output_dim, lstm_hidden_dim):
        super().__init__()
        self.handcards_fe = CardsFeatureExtraction(handcards_dim, feature_dim)
        self.action_fe = CardsFeatureExtraction(action_dim, feature_dim)
        self.attn = nn.MultiheadAttention(feature_dim, atten_head, batch_first=True)
        layers = []
        for i in range(len(dense_layers) - 1):
            layers.append(nn.Linear(dense_layers[i], dense_layers[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(dense_layers[-1], output_dim)
        lstm_input_dim = 1
        for dim in handcards_dim:
            lstm_input_dim *= dim
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, batch_first=True)
        self.flat_keys = ['handcards_num', 'gui_num', 'played_gui_num', 'played_A_num', 'is_free_play', 'is_lose']

    def forward(self, inputs, action):
        handcards = inputs['handcards']
        handcards = self.handcards_fe(handcards)

        remain_cards = inputs['remain_cards']
        remain_cards = self.handcards_fe(remain_cards)

        desk_cards = inputs['desk_cards']
        desk_shape = desk_cards.shape
        desk_cards = desk_cards.reshape((desk_shape[0] * desk_shape[1], *desk_shape[2:]))
        desk_cards = self.action_fe(desk_cards)
        desk_cards = desk_cards.reshape((*desk_shape[0:2], *desk_cards.shape[1:]))

        flat_features = []
        for key in self.flat_keys:
            features = inputs[key]
            features = features.reshape((features.shape[0], -1))
            flat_features.append(features)

        flat_features = torch.hstack(flat_features)

        action_feature = self.action_fe(action[:, 0, :])
        action_desk_cards_feature = self.action_fe(action[:, 1, :])

        features = torch.cat((handcards, remain_cards, flat_features, action_feature, action_desk_cards_feature), dim=-1)

        action_feature = action_feature.reshape((action_feature.shape[0], 1, *action_feature.shape[1:]))
        attn_output, attn_output_weights = self.attn(action_feature, desk_cards, desk_cards)
        attn_output = attn_output.reshape((attn_output.shape[0], *attn_output.shape[2:]))

        history_feature = inputs['history_actions']
        history_feature_shape = history_feature.shape
        history_feature = history_feature.reshape((*history_feature.shape[0:2], -1))
        history_feature, (h_n, _) = self.lstm(history_feature)
        history_feature = history_feature[:, -1, :]

        features = torch.cat((attn_output, features, history_feature), dim=-1)
        x = features
        for linear in self.layers:
            x = linear(x)
            x = torch.relu(x)
        y = self.linear(x)
        return y

class TransformerNetEx(nn.Module):
    def __init__(self, action_dim, atten_head, dense_layers, output_dim):
        super().__init__()
        feature_dim = 1
        for dim in action_dim:
            feature_dim *= dim
        self.action_attn = nn.MultiheadAttention(feature_dim, atten_head, batch_first=True)
        self.history_attn = nn.MultiheadAttention(feature_dim, atten_head, batch_first=True)
        layers = []
        for i in range(len(dense_layers) - 1):
            layers.append(nn.Linear(dense_layers[i], dense_layers[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(dense_layers[-1], output_dim)
        self.flat_keys = ['handcards_num', 'gui_num', 'played_gui_num', 'played_A_num', 'is_free_play', 'is_lose']

    def attention(self, action_feature, action_list, attn_net):
        action_list_shape = action_list.shape
        action_list = action_list.reshape((*action_list.shape[0:2], -1))

        action_feature = action_feature.reshape((action_feature.shape[0], 1, *action_feature.shape[1:]))
        attn_output, attn_output_weights = attn_net(action_feature, action_list, action_list)
        attn_output = attn_output.reshape((attn_output.shape[0], *attn_output.shape[2:]))
        return attn_output

    def forward(self, inputs, action):
        handcards = inputs['handcards']
        handcards = handcards.reshape((handcards.shape[0], -1))

        remain_cards = inputs['remain_cards']
        remain_cards = remain_cards.reshape((remain_cards.shape[0], -1))

        flat_features = []
        for key in self.flat_keys:
            features = inputs[key]
            features = features.reshape((features.shape[0], -1))
            flat_features.append(features)
        flat_features = torch.hstack(flat_features)

        action_feature = action[:, 0, :]
        action_desk_cards_feature = action[:, 1, :]
        action_feature = action_feature.reshape(action_feature.shape[0], -1)
        action_desk_cards_feature = action_desk_cards_feature.reshape(action_desk_cards_feature.shape[0], -1)

        features = torch.cat((handcards, remain_cards, flat_features, action_feature, action_desk_cards_feature), dim=-1)

        desk_cards = inputs['desk_cards']
        desk_attn_feature = self.attention(action_feature.clone(), desk_cards, self.action_attn)

        history_feature = inputs['history_actions']
        history_attn_feature = self.attention(action_feature.clone(), history_feature, self.history_attn)

        features = torch.cat((desk_attn_feature, features, history_attn_feature), dim=-1)
        x = features
        for linear in self.layers:
            x = linear(x)
            x = torch.relu(x)
        y = self.linear(x)
        return y
