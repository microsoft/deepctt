"""PyTorch implementation of the Wild Bootstrap Block Test (WB-Block)

This test is a combination of the Block MMD test (https://proceedings.neurips.cc/paper/2013/file/a49e9411d64ff53eccfdd09ad10a15b3-Paper.pdf)
with the Wild Bootstrap test (http://proceedings.mlr.press/v23/fromont12/fromont12.pdf).

Ported from the Cython implementation at
https://github.com/microsoft/goodpoints/blob/71c880cf354f4704d71f14f24f82dea1283532ac/examples/mmd_test/util_sqMMD_estimators.py#L245
which is based on the paper:
    Carles Domingo-Enrich and Raaz Dwivedi and Lester Mackey
    Compress Then Test: Powerful Kernel Testing in Near-linear Time
    https://arxiv.org/pdf/2301.05974.pdf
"""

from functools import partial
from typing import Callable
import torch
from deepctt.kernels import deep_gsn_kernel_single
from deepctt.ctt import get_test_results


def wild_bootstrap_block_test(
    X1: torch.Tensor,
    X2: torch.Tensor,
    B: int,
    block_size: int,
    alpha: float = 0.05,
    sigma: float = 1,
    sigma0: float = 1,
    ep: float = 0,
    d_embd: int = 20,
    starts: dict = {},
    ends: dict = {},
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wild Bootstrap Block Test

    NOTE: when X1.shape[0] == block_size, this test is equivalent to the exact test with wild bootstrap.

    Args:
        X1 (torch.Tensor): 2D array of size (n1,d)
        X2 (torch.Tensor): 2D array of size (n2,d)
        B (int): number of permutations
        block_size (int): size of each block
        alpha (float): nominal level
        sigma (float): bandwidth for the Gaussian kernel on the embeddings
        sigma0 (float): bandwidth for the Gaussian kernel on the original inputs
        ep (float): weight on the Gaussian kernel on the original inputs;
            a weight of (1-ep) is placed on the Gaussian kernel on the embeddings
        d_embd (int): number of dimensions of the embeddings
        starts (dict): dictionary of timing events
        ends (dict): dictionary of timing events

    Returns:
        h_u: hypothesis test result; 1 if null is rejected, 0 otherwise
        threshold_u: test threshold
        mmd_value_u: test statistic value

    """
    n1, d = X1.shape
    n2, d = X2.shape
    assert n1 == n2, "X1 and X2 must have the same number of samples"
    assert n1 % block_size == 0, "n must be divisible by block_size"
    n_splits = n1 // block_size

    # Step 1: Compute the block-wise h matrix
    # define kernel functions
    kernel_fn_single = partial(
        deep_gsn_kernel_single,
        sigma0=sigma0,
        sigma=sigma,
        epsilon=ep,
        d_embd=d_embd,
    )
    if starts:
        starts["time_avg_matrix"].record()
    # compute block-wise h matrix with shape (n_splits, block_size, block_size)
    h_matrix = compute_block_h_matrix(X1, X2, kernel_fn_single, n_splits, block_size)
    if ends:
        ends["time_avg_matrix"].record()

    # Step 2: Perform permutation test
    if starts:
        starts["time_perm"].record()
    epsilon = (
        2
        * torch.randint(
            2, size=(B, n_splits, block_size), dtype=X1.dtype, device=X1.device
        )
        - 1
    )
    estimator_values = torch.zeros(B + 1, device=X1.device)
    # set the first B values to the bootstrap values
    for i in range(B):
        eps_i_T = epsilon[i].view(n_splits, 1, block_size)
        eps_i = epsilon[i].view(n_splits, block_size, 1)
        estimator_values[i] = torch.bmm(torch.bmm(eps_i_T, h_matrix), eps_i).sum()
    if ends:
        ends["time_perm"].record()

    if starts:
        starts["time_stat"].record()
    estimator_values[B] = h_matrix.sum()
    if ends:
        ends["time_stat"].record()
    estimator_values /= block_size * (block_size - 1) * n_splits

    if starts:
        starts["time_test"].record()
    test_results = get_test_results(estimator_values, alpha)
    if ends:
        ends["time_test"].record()

    return test_results


def compute_block_h_matrix(
    X1: torch.Tensor,
    X2: torch.Tensor,
    kernel_fn_single: Callable,
    n_splits: int,
    block_size: int,
) -> torch.Tensor:
    """Compute the block-wise h matrix.

    Args:
        X1 (torch.Tensor): 2D array of size (n1,d)
        X2 (torch.Tensor): 2D array of size (n2,d)
        kernel_fn_single (Callable): kernel function
        n_splits (int): number of splits
        block_size (int): size of each block

    Returns:
        h_matrix (torch.Tensor): block-wise h matrix of size (n_splits, block_size, block_size)

    """
    # reshape X1, X2 to shape (n_splits, block_size, d)
    X1 = X1.view(n_splits, block_size, -1)
    X2 = X2.view(n_splits, block_size, -1)
    # concatenate X1 and X2 along the second dimension
    X = torch.cat((X1, X2), dim=1)
    # compute kernel matrix
    K = kernel_fn_single(X)
    # compute block-wise h matrix
    h_matrix = (
        K[:, :block_size, :block_size]
        + K[:, block_size:, block_size:]
        - K[:, :block_size, block_size:]
        - K[:, block_size:, :block_size]
    )
    # set the diagonals to zero so that we only sum the off-diagonal elements j =/= k
    h_matrix.diagonal(dim1=-2, dim2=-1).fill_(0)
    return h_matrix
