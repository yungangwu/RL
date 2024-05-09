from .policy_registry import register

register(id="random", entry_point="ddz.policy.random.policy_random:PolicyRandom")
register(id="minor_random", entry_point="ddz.policy.random.minor_policy_random:MinorPolicyRandom")
register(id="play_net", entry_point="ddz.policy.net.policy_play_net:PolicyNet")
register(id="minor_rule", entry_point="ddz.policy.net.minor_policy_rules:MinorPolicyRules")
register(id="minor_net", entry_point="ddz.policy.net.minor_policy_net:MinorPolicyNet")