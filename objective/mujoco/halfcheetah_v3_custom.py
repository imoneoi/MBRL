import torch


def objective(
        obs: torch.Tensor,
        act: torch.Tensor,
        next_obs: torch.Tensor
):
    # Original reward function from env

    dt = 0.05
    control_penalty = -0.1 * torch.sum(torch.square(act), dim=-1)
    running_reward = (next_obs[:, 17] - obs[:, 17]) / dt

    return -(control_penalty + running_reward)
