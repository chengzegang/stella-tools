import torch
from torch import Tensor


def sim3_to_so3(sim3: Tensor) -> Tensor:
    """
    Convert Similarity transform to Special Orthogonal transform
    """
    sR = sim3[:, :3, :3]
    t = sim3[:, :3, 3]
    if torch.linalg.det(sR) < 0:
        sR *= -1
        t *= -1
    scale = torch.linalg.det(sR) ** (1 / 3)
    R = sR / scale
    so3 = torch.zeros(sim3.shape[0], 4, 4).type_as(sim3)
    so3[:, :3, :3] = R
    so3[:, :3, 3] = t
    so3[:, 3, 3] = scale
    return so3


def reproj(x: Tensor, A: Tensor, B: Tensor) -> Tensor:
    """
    x: (B1, N, 4)
    A: (B1, 4, 4)
    B: (B2, 4, 4)
    """
    x = torch.einsum(
        "...ij,...j",
        B,
        torch.einsum("...ij,...j", A.inverse(), x),
    )
    return x
