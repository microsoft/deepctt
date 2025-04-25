"""Implementation of the KT-Compress(delta) thinning algorithm detailed in 
    Annabelle Michael Carrell, Albert Gong, Abhishek Shetty, Raaz Dwivedi, Lester Mackey
    Low-Rank Thinning
    https://arxiv.org/pdf/2502.12063
"""

import torch
from torch import (
    flip,
    linspace,
    argmin,
    logical_xor,
    zeros,
    empty,
    arange,
    log,
    tensor,
    cummax,
)
from torch import sqrt as torch_sqrt
from .kernels import deep_gsn_kernel_single


def largest_power_of_four(n: int) -> int:
    """Returns largest power of four less than or equal to n

    Args:
        n: integer
    Returns:
        largest power of four less than or equal to n

    """
    return 4 ** ((n.bit_length() - 1) // 2)


def halve_K(
    K: torch.Tensor, delta: float = 0.5, symmetrize: bool = False
) -> torch.Tensor:
    """Batched kernel halving

    NOTE: Assumes S is even.

    Args:
        K: kernel matrix with shape [B,S,S]
        delta: kernel halving is run with failure probabilities delta_i = delta/n
        symmetrize: if False, returns initial coreset for each batch;
            if True, returns initial coreset or its complement uniformly at
            random for each batch

    Returns:
        tensor of shape [B,S//2] representing the indices of the coreset points

    """
    # Extract relevant dimensions and device
    B, S, _ = K.shape
    device = K.device
    num_points_in_coreset = S // 2

    # Generate uniform(-1,1) random variables
    # NOTE: if using torch.compile with torch=2.4.0, we have to cast S to tensor
    # because torch.compile makes S a SymFloat, for which log does not seem to be defined
    log_multiplier = 0.5 + log(2 * tensor(S, device=device) / delta)

    uniforms = empty(B, num_points_in_coreset, device=device).uniform_(
        -log_multiplier, log_multiplier
    )

    # Form kernel difference matrices:
    # Kdiff[j,i] = (e_{2*j} - e_{2*j+1})^t K[b,:,:] (e_{2*i} - e_{2*i+1})
    # For each i, compute K[:, 2*i] - K[:, 2*i+1]
    Kdiff = K[:, :, ::2] - K[:, :, 1::2]
    # For each j, compute Kdiff[2*j,:] - Kdiff[2*j+1,:]
    Kdiff = Kdiff[:, ::2, :] - Kdiff[:, 1::2, :]

    # Keep track of coreset sum difference, i.e.,
    # coreset_sum_diff[i] =
    # sum_{j < i} K[2*i,coreset0[j]] - K[2*i,coreset1[j]]
    # -(sum_{j < i} K[2*i+1,coreset0[j]] - K[2*i+1,coreset1[j]])
    # Initially assign point 0 to coreset 0 and point 1 to coreset 1
    # so initial coreset_sum_diff[1] = Kdiff[0,1]
    # Note: entry stored in coreset_sum_diff[0] is irrelevant
    # and will ultimately be ignored
    coreset_sum_diff = Kdiff[:, 0, :]

    # Define thresholds
    rt_diag = torch_sqrt(Kdiff.diagonal(dim1=-2, dim2=-1))
    uniforms = uniforms * rt_diag * cummax(rt_diag, dim=-1).values

    # Add index 0 to coreset0 (and index 1 to coreset1)
    # That is, swap_points[0] = False
    swap_points = empty(B, num_points_in_coreset, device=device, dtype=torch.bool)
    swap_points[:, 0] = zeros(B, device=device, dtype=torch.bool)

    # For all remaining (2*i, 2*i+1) index pairs, add 2*i+1 to coreset0
    # if uniforms[...,i] <= coreset_sum_diff[...,i] and add 2*i otherwise
    # Intuition: more likely to swap when <x0 - x1, coreset0 - coreset1> is large
    for i in range(1, num_points_in_coreset):
        swap_points_i = uniforms[:, i] <= coreset_sum_diff[:, i]
        swap_points[:, i] = swap_points_i
        # Update cumulative coreset sum difference
        coreset_sum_diff = coreset_sum_diff + (
            1.0 - 2.0 * swap_points_i.unsqueeze(-1)
        ) * Kdiff.select(-2, i)
    if symmetrize:
        # Choose coreset 0 or 1 uniformly at random
        swap_points = logical_xor(
            swap_points, empty(1, device=device, dtype=torch.bool).bernoulli_()
        )
    # Return coreset0
    return swap_points + arange(0, S, step=2, device=device).expand_as(swap_points)


def refine_K(K: torch.Tensor, coreset: torch.Tensor) -> torch.Tensor:
    """Refinement step of the KT-Compress thinning algorithm

    Replaces each element of a coreset in turn by the input point that yields
    the greatest decrease in MMD between the resulting coreset and all input
    points (if meanK is not None) or between the resulting coreset and the
    zero measure (if meanK is None).
    Here X is implicitly represented by its kernel matrix K satisfying
    K[ii,:] = kernel(X[ii], X).

    Args:
        K: Matrix of kernel evaluations with shape [B,S,S]
        coreset: Row indices of K representing coreset [B,coreset_size]

    Returns:
        refined coreset: Row indices of K representing refined coreset [B,coreset_size]

    """
    B, S, _ = K.shape
    coreset_size = coreset.shape[-1]

    # Compute sufficient statistic representing how much each point would
    # change the quantity (coreset_size^2/2) * MMD^2(P,Q) if added into the coreset
    # with weight 1/coreset_size.
    # Since coreset_size * MMD^2(P,Q) = coreset_size * (PPk - 2QPK + QQK),
    # the impact of adding (1/coreset_size) delta_y to Q =
    #   -coreset_size delta_y PK + delta_y delta_y K / 2 + coreset_size delta_y Q K.
    # Statistic will later be adjusted to remove the influence of an
    # eliminated point

    # Initialize sufficient_stat of shape [B,S] to diagonal of K over 2
    sufficient_stat = K.diagonal(dim1=-2, dim2=-1) / 2.0

    # Construct meanK_coreset
    # First select K_coreset = K(..., :, coreset[b,a,h])
    # with shape [B,S,coreset_size]
    K_coreset = K.gather(-1, coreset.unsqueeze(-2).expand(B, S, coreset_size))
    # Take the mean over coreset dim, shape [B,S]
    meanK_coreset = K_coreset.mean(-1)
    # Update sufficient stats with diff between meanK coreset vs whole
    sufficient_stat = sufficient_stat + coreset_size * (meanK_coreset - K.mean(-1))

    # Initialize refined coreset
    refined_coreset = empty(B, coreset_size, device=coreset.device, dtype=coreset.dtype)

    # Replace each coreset element with best alternative in turn
    for coreset_idx in range(coreset_size):
        # Remove the contribution of coreset point from the normalized coreset sum in sufficient stat:
        # - delta_x delta_y K
        sufficient_stat = sufficient_stat - K_coreset[:, :, coreset_idx]
        # Replace coreset point with best point
        argmin_sufficient_stat = argmin(sufficient_stat, dim=-1)
        refined_coreset[:, coreset_idx] = argmin_sufficient_stat
        # Add influence of selected point to sufficient_stat
        sufficient_stat = sufficient_stat + K.gather(
            -1, argmin_sufficient_stat.unsqueeze(-1).unsqueeze(-1).expand(B, S, 1)
        ).squeeze(-1)
    return refined_coreset


def halve(
    X: torch.Tensor,
    sigma0: float,
    sigma: float,
    epsilon: float,
    d_embd: int,
    halve_prob: float,
    only_split: bool = True,
    refine: bool = True,
) -> torch.Tensor:
    """Batched kernel halving

    NOTE: Assumes S is even

    Args:
        X: tensor of shape [B,S,E]
        sigma0: initial kernel bandwidth
        sigma: deep embedding kernel bandwidth
        epsilon: deep kernel hyperparameter
        d_embd: deep kernel embedding dimension
        halve_prob: halve_K is run with delta = halve_prob * S^2
        only_split: if False, returns initial coreset for each batch;
            if True, returns initial coreset or its complement uniformly at
            random for each batch
        refine: if True, run refine on the output of the last halving step (see Algorithm H.1 for details)

    Returns:
        tensor of shape [B,S//2,E] representing the coreset points

    """
    B, S, E = X.shape
    # Compute failure probability parameter
    delta = halve_prob * S * S
    # Compute batched kernel matrix of shape [B,S,S]
    kernel_mat = deep_gsn_kernel_single(X, sigma0, sigma, epsilon, d_embd)
    # Identify a single shared kernel halving coreset of size S//2
    coreset = halve_K(kernel_mat, delta, symmetrize=only_split)
    if refine:
        refined_coreset = refine_K(kernel_mat, coreset)
    else:
        refined_coreset = coreset
    # Return selected half of X
    return X.gather(-2, refined_coreset.unsqueeze(-1).expand(B, S // 2, E))


def _compress(
    X: torch.Tensor,
    four_to_g_plus_1: int,
    m: int,
    sigma0: float,
    sigma: float,
    epsilon: float,
    d_embd: int,
    halve_prob: float,
    only_split: bool,
    refine: bool,
) -> torch.Tensor:
    """Helper function for the KT-Compress thinning algorithm

    Args:
        X: tensor of shape [S, E]
        four_to_g_plus_1: 4^{g+1} for the Compress oversampling parameter, g
        m: number of thinning steps
        sigma0: initial kernel bandwidth
        sigma: deep embedding kernel bandwidth
        epsilon: deep kernel hyperparameter
        d_embd: deep kernel embedding dimension
        halve_prob: KT halving probability
        only_split: if False, returns initial coreset for each batch;
            if True, returns initial coreset or its complement uniformly at
            random for each batch
        refine: if True, run refine on the output of the last halving step (see Algorithm H.1 for details)

    Returns:
        tensor of shape [S//2^m, E]

    """
    _, E = X.shape
    for i in range(m):
        bucket_size = 2**i * four_to_g_plus_1
        X = X.view(-1, bucket_size, E)
        # halve the bucket_size dimension, returning a tensor of shape [:, bucket_size//2, E]
        X = halve(
            X,
            sigma0,
            sigma,
            epsilon,
            d_embd,
            halve_prob=halve_prob,
            only_split=only_split,
            refine=False if i < m - 1 else refine,
        )
    return X.view(-1, E)


@torch.compile(mode="reduce-overhead", fullgraph=True)
def _sum_kernel_by_bin__compress(
    X: torch.Tensor,
    four_to_g_plus_1: int,
    m: int,
    sigma0: float,
    sigma: float,
    epsilon: float,
    d_embd: int,
    halve_prob: float,
    only_split: bool,
    refine: bool,
    num_bins: int,
) -> torch.Tensor:
    # Split X into `num_bins` bins, then apply KT-Compress each of these bins
    # This yields a tensor of shape [2^g sqrt(num_bins * S), E]
    X_compress = _compress(
        X,
        four_to_g_plus_1,
        m,
        sigma0,
        sigma,
        epsilon,
        d_embd,
        halve_prob,
        only_split,
        refine,
    )
    # Compute the binned kernel matrix of shape [num_bins, num_bins]
    n, d = X_compress.shape
    bin_size = n // num_bins
    return (
        deep_gsn_kernel_single(X_compress.unsqueeze(0), sigma0, sigma, epsilon, d_embd)
        .squeeze(0)
        .view(num_bins, bin_size, num_bins, bin_size)
        .transpose(1, 2)
        .sum(dim=(2, 3))
    )


def sum_kernel_by_bin__compress(
    X: torch.Tensor,
    sigma0: float,
    sigma: float,
    epsilon: float,
    d_embd: int,
    g: int = 0,
    num_bins: int = 4,
    delta: float = 0.5,
    refine: bool = True,
) -> torch.Tensor:
    """KT-Compress thinning algorithm

    Args:
        X: Tensor of shape [S, E]
        sigma0: Initial kernel bandwidth
        sigma: Deep embedding kernel bandwidth
        epsilon: Deep kernel hyperparameter
        d_embd: Deep kernel embedding dimension
        g: Oversampling factor, int >= 0
        num_bins: Number of Compress bins, int > 0
        delta: Kernel halving failure parameter, scalar in [0,1]
        refine: if True, run refine on the output of the last halving step (see Algorithm H.1 for details)

    Returns:
        Tensor of shape [2^g sqrt(num_bins * S), E]

    """
    device = X.device
    # input coreset size and data dimension, respectively
    S, E = X.shape

    # If S over num_bins is not a power of 4,
    # thin down to the nearest power of 4 using standard thinning
    # (i.e., by retaining every t-th index)
    bin_size = largest_power_of_four(S // num_bins)
    n = bin_size * num_bins
    if n != S:
        # Thin backwards from the end
        indices = flip(
            linspace(start=S - 1, end=0, steps=n, dtype=int, device=device), dims=(0,)
        )
        X = X[..., indices, :]

    # Target output size = 2^g sqrt(num_bins * n)
    # Return all indices if input size n <= target output size,
    # or equivalently, if bin_size <= 4^g
    four_to_g_plus_1 = 4 ** (g + 1)
    if bin_size < four_to_g_plus_1:
        # Number of indices is no larger than target output size
        return (
            deep_gsn_kernel_single(X.unsqueeze(0), sigma0, sigma, epsilon, d_embd)
            .squeeze(0)
            .view(num_bins, bin_size, num_bins, bin_size)
            .transpose(1, 2)
            .sum(dim=(2, 3))
        )

    # Compute base halving probability for compress
    # Note: (bin_size.bit_length() - 1)//2 = log2(sqrt(bin_size))
    # NOTE: log2_sqrt_bin_size_minus_g is the number of halving steps
    log2_sqrt_bin_size_minus_g = (bin_size.bit_length() - 1) // 2 - g
    halve_prob = delta / four_to_g_plus_1 / log2_sqrt_bin_size_minus_g / n

    # Apply KT-Compress to thin X, then compute the binned kernel matrix
    avg_matrix = _sum_kernel_by_bin__compress(
        X,
        int(four_to_g_plus_1),
        int(log2_sqrt_bin_size_minus_g),
        sigma0=sigma0,
        sigma=sigma,
        epsilon=epsilon,
        d_embd=d_embd,
        halve_prob=halve_prob,
        only_split=True,
        refine=refine,
        num_bins=num_bins,
    )
    return avg_matrix
