import torch


def sinkhorn(cost: torch.Tensor, eps: float = 0.05, iters: int = 50) -> torch.Tensor:
    batch, n, m = cost.shape
    log_k = -cost / eps
    u = torch.zeros(batch, n, device=cost.device, dtype=cost.dtype)
    v = torch.zeros(batch, m, device=cost.device, dtype=cost.dtype)
    log_mu = torch.zeros(batch, n, device=cost.device, dtype=cost.dtype) - torch.log(
        torch.tensor(float(n), device=cost.device, dtype=cost.dtype)
    )
    log_nu = torch.zeros(batch, m, device=cost.device, dtype=cost.dtype) - torch.log(
        torch.tensor(float(m), device=cost.device, dtype=cost.dtype)
    )

    for _ in range(iters):
        u = log_mu - torch.logsumexp(log_k + v[:, None, :], dim=-1)
        v = log_nu - torch.logsumexp(log_k + u[:, :, None], dim=-2)

    log_p = log_k + u[:, :, None] + v[:, None, :]
    return torch.exp(log_p)
