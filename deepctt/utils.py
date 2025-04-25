"""Functionality for training the deep kernel.

Adapted from the original code at:
- https://github.com/fengliu90/DK-for-TST/blob/master/Deep_Kernel_HIGGS.py
- License: MIT
- Copyright (c) 2021 Feng Liu
- Paper reference:
    Feng Liu, Wenkai Xu, Jie Lu, Guangquan Zhang, Arthur Gretton, Danica J. Sutherland
    Learning Deep Kernels for Non-Parametric Two-Sample Tests
    https://arxiv.org/pdf/2002.09116
"""

import numpy as np
import torch
from tqdm import tqdm
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ModelLatentF(torch.nn.Module):
    """define deep networks."""

    def __init__(self, x_in: int, H: int, x_out: int) -> None:
        """Init latent features

        Args:
            x_in (int): number of neurons in the input layer
            H (int): number of neurons in the hidden layer
            x_out (int): number of neurons in the output layer

        """
        super(ModelLatentF, self).__init__()
        self.restored = False

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward the LeNet

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor

        """
        fealant = self.latent(input)
        return fealant


def MatConvert(x: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert the numpy to a torch tensor

    Args:
        x (np.ndarray): numpy array
        device (torch.device): device to run the training on
        dtype (torch.dtype): data type to use for the training

    Returns:
        torch.Tensor: output tensor

    """
    x = torch.from_numpy(x).to(device, dtype)
    return x


def Pdist2(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the paired distance between x and y.

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor

    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return Pdist


def h1_mean_var_gram(
    Kx: torch.Tensor,
    Ky: torch.Tensor,
    Kxy: torch.Tensor,
    is_var_computed: bool,
    use_1sample_U: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Compute value of MMD and std of MMD using kernel matrix.

    Args:
        Kx (torch.Tensor): kernel matrix
        Ky (torch.Tensor): kernel matrix
        Kxy (torch.Tensor): kernel matrix
        is_var_computed (bool): whether to compute the variance of MMD
        use_1sample_U (bool): whether to use the one-sample U-statistic

    Returns:
        mmd2 (torch.Tensor): MMD value
        varEst (torch.Tensor | None): variance of MMD
        Kxyxy (torch.Tensor): kernel matrix

    """
    Kxxy = torch.cat((Kx, Kxy), 1)
    Kyxy = torch.cat((Kxy.transpose(0, 1), Ky), 1)
    Kxyxy = torch.cat((Kxxy, Kyxy), 0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div(
                (torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1))
            )
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None, Kxyxy
    hh = Kx + Ky - Kxy - Kxy.transpose(0, 1)
    V1 = torch.dot(hh.sum(1) / ny, hh.sum(1) / ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4 * (V1 - V2**2)
    return mmd2, varEst, Kxyxy


def MMDu(
    Fea: torch.Tensor,
    len_s: int,
    Fea_org: torch.Tensor,
    sigma: float,
    sigma0: float = 0.1,
    epsilon: float = 10 ** (-10),
    is_smooth: bool = True,
    is_var_computed: bool = True,
    use_1sample_U: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Compute value of deep-kernel MMD and std of deep-kernel MMD using merged data.

    Args:
        Fea (torch.Tensor): input tensor
        len_s (int): length of the input tensor
        Fea_org (torch.Tensor): input tensor
        sigma (float): bandwidth for the Gaussian kernel on the embeddings
        sigma0 (float): bandwidth for the Gaussian kernel on the original inputs
        epsilon (float): weight on the Gaussian kernel on the embeddings
        is_smooth (bool): whether to use the smooth kernel
        is_var_computed (bool): whether to compute the variance of MMD
        use_1sample_U (bool): whether to use the one-sample U-statistic

    Returns:
        mmd2 (torch.Tensor): MMD value
        varEst (torch.Tensor | None): variance of MMD
        Kxyxy (torch.Tensor): kernel matrix

    """
    X = Fea[0:len_s, :]  # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :]  # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :]  # fetch the original sample 1
    Y_org = Fea_org[len_s:, :]  # fetch the original sample 2
    L = 1  # generalized Gaussian (if L>1)

    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    if is_smooth:
        Kx = (1 - epsilon) * torch.exp(
            -((Dxx / sigma0) ** L) - Dxx_org / sigma
        ) + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1 - epsilon) * torch.exp(
            -((Dyy / sigma0) ** L) - Dyy_org / sigma
        ) + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1 - epsilon) * torch.exp(
            -((Dxy / sigma0) ** L) - Dxy_org / sigma
        ) + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)

    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)


def train_deep_kernel(
    s1: np.ndarray,
    s2: np.ndarray,
    N_epoch: int,
    device: torch.device,
    dtype: torch.dtype,
    input_dim: int,
    learning_rate: float,
    hidden_dim: int,
    embedding_dim: int,
) -> Tuple[torch.nn.Module, float, float, float]:
    """Train the deep kernel to maximize test power.

    Args:
        s1 (np.ndarray): input tensor of shape (n1,d)
        s2 (np.ndarray): input tensor of shape (n2,d)
        N1 (int): number of samples in the first dataset
        N_epoch (int): number of training epochs
        device (torch.device): device to run the training on
        dtype (torch.dtype): data type to use for the training
        input_dim (int): dimension of the data
        learning_rate (float): learning rate for the training
        hidden_dim (int): number of neurons in the hidden layer
        embedding_dim (int): number of neurons in the embedding layer

    Returns:
        model_u (torch.nn.Module): trained deep kernel model
        sigma0_u (float): bandwidth for the Gaussian kernel on the original inputs
        sigma (float): bandwidth for the Gaussian kernel on the embeddings
        ep (float): weight on the Gaussian kernel on the embeddings

    """
    J_star_u = np.zeros([N_epoch])
    # Initialize parameters
    model_u = ModelLatentF(input_dim, hidden_dim, embedding_dim).to(device)
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))  # noqa: N816
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * input_dim), device, dtype)  # noqa: N816
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)  # noqa: N816
    sigma0OPT.requires_grad = False
    logger.debug(f"epsilonOPT: {epsilonOPT.item()}")
    logger.debug(f"sigmaOPT: {sigmaOPT.item()}")
    logger.debug(f"sigma0OPT: {sigma0OPT.item()}")

    # Setup optimizer for training deep kernel
    optimizer_u = torch.optim.Adam(
        list(model_u.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT],
        lr=learning_rate,
    )

    N1 = len(s1)
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)

    logger.info("Training deep kernel to maximize test power...")
    pbar = tqdm(range(N_epoch))
    for t in pbar:
        # Compute epsilon, sigma and sigma_0
        ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
        sigma = sigmaOPT**2
        sigma0_u = sigma0OPT**2
        # Compute output of the deep network
        modelu_output = model_u(S)
        # Compute J (STAT_u)
        mmd2, mmd_std, _ = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep)
        mmd_value_temp = -1 * (mmd2 + 10 ** (-8))
        mmd_std_temp = torch.sqrt(mmd_std + 10 ** (-8))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        J_star_u[t] = STAT_u.item()
        # Initialize optimizer and Compute gradient
        optimizer_u.zero_grad()
        STAT_u.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_u.step()
        pbar.set_description(f"Statistic (higher is better): {-J_star_u[t]:.4f}")

    # Kernel Parameters
    # NOTE: in the original DK-for-TST code, sigma0_u, sigma and ep are tensors of shape (1,)
    # but here we detach epsilon, sigma and sigma_0 and extract scalar
    sigma0_u = sigma0_u.detach()[0]
    sigma = sigma.detach()[0]
    ep = ep.detach()[0]

    return model_u, sigma0_u, sigma, ep
