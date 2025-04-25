"""Power and runtime script for two-sample tests on the Higgs dataset.

This script reproduces the Higgs experiment from Sec. 6.2 of the following paper:
    Annabelle Michael Carrell, Albert Gong, Abhishek Shetty, Raaz Dwivedi, Lester Mackey
    Low-Rank Thinning
    https://arxiv.org/pdf/2502.12063

We adapted the following script:
- https://github.com/fengliu90/DK-for-TST/blob/master/Deep_Kernel_HIGGS.py
- License: MIT
- Copyright (c) 2021 Feng Liu
- Paper reference:
    Feng Liu, Wenkai Xu, Jie Lu, Guangquan Zhang, Arthur Gretton, Danica J. Sutherland
    Learning Deep Kernels for Non-Parametric Two-Sample Tests
    https://arxiv.org/pdf/2002.09116

NOTE: we were not able to reproduce the results of Liu et al. exactly because
different pytorch versions were used.

Example usage:
```bash
python run_power.py -n 4096 --method ctt --g 0 --kk 0
```
"""

import numpy as np
import pandas as pd
import torch
import os
import logging
from tqdm import tqdm
from typing import Dict, Optional, Tuple

# import Exact Test
from utils_HD import MatConvert, MMDu, ModelLatentF

# import CTT Test
from deepctt.ctt import ctt

# import Wild Bootstrap Test baseline
from wb_block import wild_bootstrap_block_test

from util_experiments import get_base_parser, generate_dist_higgs

parser = get_base_parser()
parser.add_argument(
    "--data_path",
    "-dp",
    type=str,
    default="data",
    help="Data path containing HIGGS_TST.pckl",
)
parser.add_argument(
    "--p_poisoning",
    type=float,
    default=0,
    help="For each sample in class 1, flip a coin with this probability to decide if it should replaced with a sample from class 0"
    "p_poisoning = 0 means we use the original data samples from class 1",
)
parser.add_argument(
    "--d", "-d", type=int, default=4, help="Number of features to consider"
)

args = parser.parse_args()
n = args.n  # number of samples in one set (power of 4)
method = args.method
g = args.g
kk = args.kk
output_path = args.output_path
data_path = args.data_path
verbose = args.verbose
p_poisoning = args.p_poisoning
d = args.d

torch.set_float32_matmul_precision("high")
torch.cuda.empty_cache()

if verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

results_dir = os.path.join(
    output_path,
    "results",
)
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, f"{method}-n{n}-g{g}-kk{kk}.csv")
if os.path.exists(results_path):
    logger.info(f"Results file {results_path} already exists. Exiting.")
    exit()

# Setup seeds
np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True
is_cuda = True
# Setup for experiments
dtype = torch.float
device = torch.device("cuda:0")
N_per = 100  # permutation times
alpha = 0.05  # test threshold
logging.info("n: " + str(n) + " d: " + str(d))
N_epoch = 1000  # number of training epochs
x_in = d  # number of neurons in the input layer, i.e., dimension of data
H = 20  # number of neurons in the hidden layer
x_out = 20  # number of neurons in the output layer
learning_rate = 0.00005
learning_ratea = 0.001
learning_rate_C2ST = 0.001  # noqa: N816
K = 10  # number of trails
N = 100  # number of test sets
N_f = 100.0  # number of test sets (float)

logger.info("Generating data...")
dataX, dataY = generate_dist_higgs(  # noqa: N816
    higgs_path=os.path.join(data_path, "HIGGS_TST.pckl"),
    p_poisoning=p_poisoning,
    rng=np.random.default_rng(1102),
)
logger.info(f"dataX size: {dataX.shape}, dataY size: {dataY.shape}")

# Naming variables
J_star_u = np.zeros([N_epoch])
J_star_adp = np.zeros([N_epoch])

torch.manual_seed(kk * 19 + n)
torch.cuda.manual_seed(kk * 19 + n)
# Initialize parameters
if is_cuda:
    model_u = ModelLatentF(x_in, H, x_out).cuda()
else:
    model_u = ModelLatentF(x_in, H, x_out)

epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))  # noqa: N816
epsilonOPT.requires_grad = True
sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * d), device, dtype)  # noqa: N816
sigmaOPT.requires_grad = True
sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)  # noqa: N816
sigma0OPT.requires_grad = False
logging.debug(epsilonOPT.item())

# Setup optimizer for training deep kernel
optimizer_u = torch.optim.Adam(
    list(model_u.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT],
    lr=learning_rate,
)


# Generate Higgs (P,Q)
N1_T = dataX.shape[0]
N2_T = dataY.shape[0]
np.random.seed(seed=1102 * kk + n)
ind1 = np.random.choice(N1_T, n, replace=False)
np.random.seed(seed=819 * kk + n)
ind2 = np.random.choice(N2_T, n, replace=False)
s1 = dataX[ind1, :d]
s2 = dataY[ind2, :d]
N1 = n
N2 = n
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
    TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep)
    mmd_value_temp = -1 * (TEMP[0] + 10 ** (-8))
    mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
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


def run_method(
    method: str,
    S: torch.Tensor,
    S_embd: torch.Tensor,
    N1: int,
    N_per: int,
    sigma: torch.Tensor,
    sigma0_u: torch.Tensor,
    ep: torch.Tensor,
    alpha: float,
    g: Optional[int] = None,
    starts: Dict[str, torch.cuda.Event] = {},
    ends: Dict[str, torch.cuda.Event] = {},
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Wrapper for the different two-sample tests.

    Args:
        method (str): one of 'exact', 'ctt', 'wb_block', 'subsampling'
        S (torch.Tensor): original data
        S_embd (torch.Tensor): deep embeddings
        N1 (int): number of samples in the first set
        N_per (int): number of permutations
        sigma (torch.Tensor): sigma
        sigma0_u (torch.Tensor): sigma0_u
        ep (torch.Tensor): epsilon
        alpha (float): significance level
        g (Optional[int]): thinning parameter
        starts (Dict[str, torch.cuda.Event]): timing events
        ends (Dict[str, torch.cuda.Event]): timing events

    Returns:
        Tuple[int, torch.Tensor, torch.Tensor]: h, threshold, mmd_value

    """
    if method == "ctt":
        # concatenate the deep embeddings with the original data
        S_embd_orig = torch.cat((S_embd, S), dim=1)

        # run ctt test
        h_u, threshold_u, mmd_value_u = ctt(
            X1=S_embd_orig[:N1],
            X2=S_embd_orig[N1:],
            g=g,
            B=N_per,
            # deep kernel parameters
            sigma0=sigma0_u,
            sigma=sigma,
            ep=ep,
            d_embd=x_out,
            alpha=alpha,
            # timing events
            starts=starts,
            ends=ends,
            refine=True,
        )
    elif method == "wb_block":
        # concatenate the deep embeddings with the original data
        S_embd_orig = torch.cat((S_embd, S), dim=1)

        # run wb-block test
        h_u, threshold_u, mmd_value_u = wild_bootstrap_block_test(
            X1=S_embd_orig[:N1],
            X2=S_embd_orig[N1:],
            B=N_per,
            block_size=int(
                N1 // 2**g
            ),  # use g as the thinning parameter rather than the oversampling parameter
            alpha=alpha,
            # deep kernel parameters
            sigma=sigma,
            sigma0=sigma0_u,
            ep=ep,
            d_embd=x_out,
            # timing events
            starts=starts,
            ends=ends,
        )
    elif method == "subsampling":
        S_embd_orig = torch.cat((S_embd, S), dim=1)
        # thin S_embd_orig to size len(S_embd_orig) // 2^g
        N1_thin = int(
            N1 / 2**g
        )  # first compute N1_thin to account for rounding errors when g is a float
        N2_thin = int((len(S_embd_orig) - N1) / 2**g)
        S_embd_orig = torch.cat((S_embd_orig[:N1_thin], S_embd_orig[N1 : N1 + N2_thin]))
        N1 = N1_thin
        assert S_embd_orig.shape[0] == 2 * N1, (
            f"S_embd_orig.shape[0] = {S_embd_orig.shape[0]} and N1 = {N1}"
        )
        # run wb-block test
        h_u, threshold_u, mmd_value_u = wild_bootstrap_block_test(
            X1=S_embd_orig[:N1],
            X2=S_embd_orig[N1:],
            B=N_per,
            block_size=N1,
            alpha=alpha,
            # deep kernel parameters
            sigma=sigma,
            sigma0=sigma0_u,
            ep=ep,
            d_embd=x_out,
            # timing events
            starts=starts,
            ends=ends,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return h_u, threshold_u, mmd_value_u


logger.info("Performing warm up...")
for _ in range(10):
    with torch.no_grad():
        S_embd = model_u(S)
        h_u, threshold_u, mmd_value_u = run_method(
            method, S, S_embd, N1, N_per, sigma, sigma0_u, ep, alpha, g=g
        )


logger.info("Computing test power of deep kernel based MMD...")
results = []

np.random.seed(1102)
count_u = 0
pbar = tqdm(range(N))
for k in pbar:
    torch.compiler.cudagraph_mark_step_begin()
    # Generate Higgs (P,Q)
    np.random.seed(seed=1102 * (k + 1) + n)
    ind1 = np.random.choice(N1_T, n, replace=False)
    np.random.seed(seed=819 * (k + 2) + n)
    ind2 = np.random.choice(N2_T, n, replace=False)
    s1 = dataX[ind1, :d]
    s2 = dataY[ind2, :d]
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    # compute deep features
    S_embd = model_u(S)

    # timing events
    starts = {}
    ends = {}
    timing_blocks = [
        "time",  # total time
        "time_compress",
        "time_concat",
        "time_avg_matrix",
        "time_perm",
        "time_stat",
        "time_test",
    ]
    for name in timing_blocks:
        starts[name] = torch.cuda.Event(enable_timing=True)
        ends[name] = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        starts["time"].record()
        h_u, threshold_u, mmd_value_u = run_method(
            method,
            S,
            S_embd,
            N1,
            N_per,
            sigma,
            sigma0_u,
            ep,
            alpha,
            g=g,
            starts=starts,
            ends=ends,
        )
        ends["time"].record()
        torch.cuda.synchronize()

    # get values of h, threshold, and MMD
    h_u = h_u.item()
    threshold_u = threshold_u.item()
    mmd_value_u = mmd_value_u.item()
    logger.debug("h: %d, Threshold: %f, MMD_value: %f", h_u, threshold_u, mmd_value_u)

    # Gather results
    count_u = count_u + h_u
    pbar.set_description(
        f"Test Power (higher is better): {count_u / (k + 1):.4f} ({count_u}/{k + 1})"
    )

    result = {
        "method": method,
        "g": g,
        "kk": kk,
        "H": h_u,
        "threshold": threshold_u,
        "mmd_value": mmd_value_u,
    }
    for name in timing_blocks:
        try:
            elapsed_time = starts[name].elapsed_time(ends[name])
        except RuntimeError:
            elapsed_time = np.nan
        result[name] = elapsed_time
        logger.debug(f"Time ({name}): {elapsed_time:.4f} ms")
    results.append(result)

df = pd.DataFrame(data=results)
logger.info(f"Test Power: {np.mean(df['H'].values): .4f}")
logger.info(f"Time per test (ms): {np.mean(df['time'].values): .4f}")

logger.info("Getting timings for specific blocks...")
# get columns starting with `time`
df["time_avg_matrix_plus_stat"] = df.apply(
    lambda row: row["time_avg_matrix"] + row["time_stat"], axis=1
)
timing_cols = [col for col in df.columns if col.startswith("time")]
print(df[timing_cols].describe())

# save results
logger.info(f"Saving results to {results_path}")
df.to_csv(results_path, index=False)
