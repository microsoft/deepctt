"""Implementation of Deep Kernel Compress-Then-Test."""

import torch
import math
from .compress import sum_kernel_by_bin__compress
import logging

logger = logging.getLogger(__name__)


def get_num_bins(n1: int, n2: int, s: int) -> tuple[int, int, int, int]:
    """Calculate the number of bins and bin sizes for the given sample sizes and parameter s.

    Args:
        n1 (int): Sample size of the first dataset.
        n2 (int): Sample size of the second dataset.
        s (int): Parameter to determine the number of bins.

    Returns:
        num_bins_total (int): Total number of bins.
        bin_size (int): Size of each bin.
        num_bins1 (int): Number of bins for the first dataset.
        num_bins2 (int): Number of bins for the second dataset.

    """
    num_bins_total = min(2 * s, n1 + n2)
    bin_size = (n1 + n2) // num_bins_total
    num_bins1 = n1 // bin_size
    num_bins2 = num_bins_total - num_bins1
    return num_bins_total, bin_size, num_bins1, num_bins2


def ctt(
    X1: torch.Tensor,
    X2: torch.Tensor,
    g: int,
    B: int = 39,
    s: int = 16,
    alpha: float = 0.05,
    delta: float = 0.5,
    sigma0: float = 1,
    sigma: float = 1,
    ep: float = 0,
    d_embd: int = 20,
    starts: dict = {},
    ends: dict = {},
    refine: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress-Then-Test Two-Sample Test using Deep Kernel

    The deep kernel is of the form
        k(x,y) = [(1-ep) k(x[:d_embd],y[:d_embd]) + ep] * q(x[d_embd:],y[d_embd:])
    where k and q are Gaussian kernels with bandwidth sigma and sigma0, respectively.

    NOTE: The current implementation assumes that len(X1)+len(X2) is num_bins x (power of 4).
    To ensure this, please thin X1 and X2 before calling this function.

    Args:
        X1 (torch.Tensor): 2D array of size (n1,d)
        X2 (torch.Tensor): 2D array of size (n2,d)
        g (int): compression level; must be >= 0
        B (int): number of permutations
        s (int): total number of compression bins will be num_bins = min(2*s, n1+n2);
            X1 will be divided into num_bins * n1 / (n1 + n2) compression bins;
            X2 will be divided into num_bins * n2 / (n1 + n2) compression bins
        alpha (float): nominal level
        delta (float): KT-Compress failure probability
        sigma0 (float): kernel bandwidth for original data
        sigma (float): kernel bandwidth for compressed data
        ep (float): hyperparameter for deep GSN kernel
        d_embd (int): embedding dimension for deep GSN kernel
        starts (dict): dictionary to store start times for each step
        ends (dict): dictionary to store end times for each step
        refine (bool): if True, perform the refinement step detailed in Algorithm H.1 of the paper

    Returns:
        h_u: hypothesis test result; 1 if null is rejected, 0 otherwise
        threshold_u: test threshold
        mmd_value_u: test statistic value

    """
    # Number of sample points
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Number of KT-Compress bins per dataset
    num_bins_total, bin_size, num_bins1, num_bins2 = get_num_bins(n1, n2, s)
    if n1 != num_bins1 * bin_size:
        logger.warning(
            f"{n1} != {num_bins1} * {bin_size}. "
            f"Please thin X1 to a multiple of {bin_size} before calling ctt."
        )
    if n2 != num_bins2 * bin_size:
        logger.warning(
            f"{n2} != {num_bins2} * {bin_size}. "
            f"Please thin X2 to a multiple of {bin_size} before calling ctt."
        )

    # Compress X1 and X2 simultaneously, then compute the binned kernel matrix
    avg_matrix = sum_kernel_by_bin__compress(
        torch.cat([X1, X2], dim=0),
        sigma0,
        sigma,
        ep,
        d_embd,
        g=g,
        num_bins=num_bins_total,
        delta=delta,
        refine=refine,
    )

    # Compute permutations
    estimator_values = torch.empty(B + 1, device=X1.device)

    if starts:
        starts["time_perm"].record()
    perm_signs = torch.rand(
        B, num_bins_total, dtype=avg_matrix.dtype, device=avg_matrix.device
    ).argsort(dim=1)
    # Then assign a +/-1 sign to each datapoint indicating if
    # bin was assigned a permutation index < num_bins1
    perm_signs = (perm_signs < num_bins1) * 2 - 1  # shape (B+1, num_bins_total)
    perm_signs = perm_signs.to(dtype=avg_matrix.dtype)
    # use for-loop so that it's comparable with wb-block implementation
    # set the first B values to the bootstrap values
    for i in range(B):
        perm_T = perm_signs[i].view(1, num_bins_total)
        perm = perm_signs[i].view(num_bins_total, 1)
        estimator_values[i] = perm_T @ avg_matrix @ perm
    if ends:
        ends["time_perm"].record()

    # Compute test statistic
    original_signs = torch.arange(
        num_bins_total, dtype=avg_matrix.dtype, device=avg_matrix.device
    )
    original_signs = (original_signs < num_bins1) * 2 - 1  # shape (B+1, num_bins_total)
    original_signs = original_signs.to(dtype=avg_matrix.dtype)
    if starts:
        starts["time_stat"].record()
    estimator_values[B] = (
        original_signs.view(1, num_bins_total)
        @ avg_matrix
        @ original_signs.view(num_bins_total, 1)
    )
    if ends:
        ends["time_stat"].record()
    estimator_values /= num_bins1 * num_bins2

    # Compute test results
    if starts:
        starts["time_test"].record()
    test_results = get_test_results(estimator_values, alpha)
    if ends:
        ends["time_test"].record()
    return test_results


def get_test_results(
    estimator_values: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute whether to reject the null hypothesis.

    Args:
        estimator_values: array of size (B+1) containing the test statistic values
        alpha: test level
    Returns:
        rejects: 1 if null is rejected, 0 otherwise
        threshold_values: test threshold
        statistic_values: test statistic value

    """
    # Extract original data test statistic
    statistic_values = estimator_values[-1]
    B_plus_1 = estimator_values.shape[0]
    # Note: we include -1 because indices go from 0 to B instead of 1 to B+1
    thresh_index = math.ceil((1 - alpha) * B_plus_1) - 1
    # sort estimator_values in place
    sorted_estimator_values, _ = torch.sort(estimator_values)

    # Identify the test statistic threshold / critical value
    threshold_values = sorted_estimator_values[thresh_index]
    if statistic_values > threshold_values:
        # Always reject
        rejects = torch.tensor(1)
    elif statistic_values == threshold_values:
        # Count the number of values > threshold
        num_greater = (
            sorted_estimator_values[thresh_index + 1 :] > threshold_values
        ).sum()
        # Count the number of values < threshold
        num_less = (
            sorted_estimator_values[: thresh_index - 1] < threshold_values
        ).sum()
        # Reject a particular fraction of the time to ensure test has level alpha
        rejects = (B_plus_1 * alpha - num_greater) / (B_plus_1 - num_greater - num_less)
    else:
        # Never reject
        rejects = torch.tensor(0)
    return rejects, threshold_values, statistic_values


def signed_matrix_sum(K: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Returns sum_{i,j} K[i,j] signs[i] signs[j]

    Args:
      K: symmetric matrix of size (n,n) (only lower triangular part is used)
      signs: vector of length n containing +/-1 values

    """
    return signs.view(1, -1) @ K @ signs.view(-1, 1)
