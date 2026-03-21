from __future__ import annotations

from torch import Tensor
import torch


def normalize_advantages(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    rewards = rewards.float()
    if rewards.numel() == 0:
        return rewards
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if torch.isclose(std, torch.zeros_like(std), atol=eps):
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + eps)


def masked_mean(values: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    v = values.float()
    m = mask.float()
    denom = m.sum().clamp(min=eps)
    return (v * m).sum() / denom


def compute_kl_penalty(student_lp: Tensor, ref_lp: Tensor, mask: Tensor) -> Tensor:
    # KL proxy used by GRPO-style regularization over sampled tokens.
    return masked_mean(student_lp.float() - ref_lp.float(), mask)


def top_k_kl_approx(
    student_logprobs_topk: Tensor,  # (..., K)
    teacher_logprobs_topk: Tensor,  # (..., K)
    eps: float = 1e-8,
) -> Tensor:
    p = student_logprobs_topk.float().exp()
    q = teacher_logprobs_topk.float().exp().clamp(min=eps)
    p_sum = p.sum(dim=-1).clamp(max=1.0)
    q_sum = q.sum(dim=-1).clamp(max=1.0)
    kl_topk = (p * (student_logprobs_topk.float() - teacher_logprobs_topk.float())).sum(dim=-1)

    tail_p = (1.0 - p_sum).clamp(min=eps)
    tail_q = (1.0 - q_sum).clamp(min=eps)
    tail_kl = tail_p * torch.log(tail_p / tail_q)
    return kl_topk + tail_kl


def jensen_shannon_div(p_logits: Tensor, q_logits: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    p_log = torch.log_softmax(p_logits.float(), dim=dim)
    q_log = torch.log_softmax(q_logits.float(), dim=dim)
    p = p_log.exp()
    q = q_log.exp()
    m = 0.5 * (p + q)
    m = m.clamp(min=eps)
    m_log = m.log()
    kl_pm = (p * (p_log - m_log)).sum(dim=dim)
    kl_qm = (q * (q_log - m_log)).sum(dim=dim)
    return 0.5 * (kl_pm + kl_qm)


def apply_ppo_clip(ratio: Tensor, advantage: Tensor, clip_low: float, clip_high: float) -> Tensor:
    # Asymmetric clipping: tighter cap for positive-advantage updates.
    lower = 1.0 - float(clip_low)
    upper = 1.0 + float(clip_high)
    clipped = ratio.clamp(min=lower, max=upper)
    return torch.where(advantage > 0, torch.minimum(ratio, clipped), torch.maximum(ratio, clipped))


if __name__ == "__main__":
    r = torch.tensor([1.0, 2.0, 3.0])
    a = normalize_advantages(r)
    assert a.shape == r.shape
    assert torch.isclose(a.mean(), torch.tensor(0.0), atol=1e-5)

    r_same = torch.tensor([2.0, 2.0, 2.0])
    assert torch.allclose(normalize_advantages(r_same), torch.zeros_like(r_same))
    print("algo/utils.py self-test passed")

