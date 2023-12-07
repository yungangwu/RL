import torch
import torch.nn as nn
import numpy as np
from continues_A3C import ACNet


def set_model_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0, std=0.1)  # 以0为均值，0.1为标准差的标准正态分布
        nn.init.constant_(layer.bias, 0.)  # 以常数0开始的偏执初始化


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def push_and_pull(gnet: ACNet, lnet: ACNet, bs, ba, br, bs_):
    bs = v_wrap(np.vstack(bs))
    ba = v_wrap(np.array(ba),
                dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(
                    np.vstack(ba))
    br = v_wrap(np.vstack(br))
    bs_ = v_wrap(np.vstack(bs_))

    loss = lnet.loss_func(bs, ba, br, bs_)

    gnet.actor_optimizer.zero_grad()
    gnet.critic_optimizer.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.actor.actor_net.parameters(),
                      gnet.actor.actor_net.parameters()):
        gp._grad = lp.grad

    for lp, gp in zip(lnet.critic.critic_net.parameters(),
                      gnet.critic.critic_net.parameters()):
        gp._grad = lp.grad

    gnet.actor_optimizer.step()
    gnet.critic_optimizer.step()

    lnet.actor.actor_net.load_state_dict(gnet.actor.actor_net.state_dict())
    lnet.critic.critic_net.load_state_dict(gnet.critic.critic_net.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:",
        global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )