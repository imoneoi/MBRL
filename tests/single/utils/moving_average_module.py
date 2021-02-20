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
        batch_size: int = 1024,
        dims: int = 10
):
    module = MovingAverageModule(dims, device=torch.device("cpu"))
    x = torch.rand((n, dims)) * max_v

