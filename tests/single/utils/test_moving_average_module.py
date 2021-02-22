import pytest
import torch

from utils.moving_average_module import MovingAverageModule


@pytest.mark.parametrize("n, max_v", [
    (10000, 1),
    (10000, 10),
    (10000, 1000),
    (10000, 100000),
])
def test_moving_average_std(
        n: int,
        max_v: float,

        batch_factor: float = 0.1,
        dims: int = 10,

        seed: int = 1
):
    # random seed
    torch.manual_seed(seed)

    # generate data
    module = MovingAverageModule(dims, device=torch.device("cpu"))
    x = -max_v + torch.rand((n, dims)) * 2 * max_v

    idx = 0
    while idx < n:
        # get random batch
        sz = torch.randint(low=1, high=max(1, int((n - idx) * batch_factor)) + 1, size=[1])
        batch = x[idx: idx + sz]
        idx += sz

        # update
        module.update(batch)

        # verify
        mean, std = module.get_mean_std()

        gt_mean = torch.mean(x[:idx], dim=0)
        gt_std = torch.sqrt(torch.mean((x[:idx] - gt_mean) ** 2, dim=0))

        a_tol = 1e-8
        r_tol = 1e-4
        assert torch.isclose(mean, gt_mean, atol=a_tol, rtol=r_tol).all()
        assert torch.isclose(std, gt_std, atol=a_tol, rtol=r_tol).all()
