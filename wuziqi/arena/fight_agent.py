import os
from util.common import get_logger
from policy.network import ACNet
from policy.ppo_policy import PPOPolicy

logger = get_logger(__name__)

class FightAgent:
    def __init__(self, net_config: str, checkpoint_path: str, device: str) -> None:
        super().__init__()
        self._device = device
        self.net = ACNet(**net_config).to(device=device)
        self.old_net = ACNet(**net_config).to(device=device)
        self.net.eval()
        self.old_net.eval()
        self.policy = PPOPolicy(self.net, self.old_net, None)
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f'load model from file [{checkpoint_path}].')
            self.policy.load(checkpoint_path)
        else:
            logger.error(f'checkpoint file [{checkpoint_path}] not found.')

    def act(self, obs: Batch, legal_actions: list):
        num_act = len(legal_actions)
        if num_act > 1:
            obs['act'] = Batch(hero_name=[0], hero_attr=0)
            batch_obs = _create_value(obs, num_act)
            for i in range(num_act):
                obs['act'] = Batch(hero_name=legal_actions[i][0], hero_attr=legal_actions[i][1])
                batch_obs[i] = obs
            batch_obs.to_torch(dtype=torch.float32, device=self._device)
            with torch.no_grad():
                q, _ = self.net(batch_obs)
            a = torch.argmax(q.flatten())
        return a