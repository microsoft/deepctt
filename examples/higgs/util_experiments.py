from argparse import ArgumentParser
import numpy as np
import logging
import pickle
from typing import Tuple


def get_base_parser() -> ArgumentParser:
    """Get the base parser for the experiments.

    Returns:
        ArgumentParser: the base parser

    """
    parser = ArgumentParser()
    parser.add_argument(
        "--n",
        "-n",
        type=int,
        default=4096,
        help="Number of samples in X and Y each",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="exact",
        help="Kernel two-sample test",
        choices=["exact", "wb_block", "subsampling", "ctt"],
    )
    parser.add_argument(
        "--g",
        "-g",
        type=float,
        default=0,
        help="Oversampling parameter for KT-Compress, for which g must be an integer,"
        "or thinning factor for subsampling, for which g can be a float",
    )
    parser.add_argument(
        "--kk",
        "-kk",
        type=int,
        default=0,
        help="Trial number (also determines the random seeds)",
    )
    parser.add_argument(
        "--output_path",
        "-op",
        type=str,
        default="out",
        help="Output path for the results",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    return parser


def generate_dist_higgs(
    higgs_path: str, p_poisoning: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates two empirical distributions P, Q from the Higgs dataset.

    P is the data from class 0 and Q is sampled from class 0 with probability `p_poisoning` and
    class 1 with probability `1-p_poisoning`. When `p_poisoning=0`, P is the data from class 0
    and Q is the data from class 1.

    Based on: https://github.com/microsoft/goodpoints/blob/ebab90308b1dbfae9608e43972d56b684164dcd2/examples/mmd_test/util_sampling.py#L126

    Args:
        higgs_path (str): the path to the Higgs dataset
        p_poisoning (float): the probability of poisoning
        rng (np.random.Generator): the random number generator

    Returns:
        Tuple[np.ndarray, np.ndarray]: P, Q

    """
    # Load data
    logging.info(f"Loading HIGGS data from {higgs_path}...")
    with open(higgs_path, "rb") as f:
        data = pickle.load(f)
    data0 = data[0]
    data1 = data[1]

    if p_poisoning == 0:
        logging.info("No poisoning.")
        P = data0
        Q = data1
    elif 0 < p_poisoning <= 1:
        P = data0  # P remains the same
        n0, n1 = data0.shape[0], data1.shape[0]
        # compute the number of poisoned samples by flipping a coin with probability p_poisoning for each sample in data1
        n_poisoned = rng.binomial(n1, p_poisoning)
        logging.info(f"Poisoning {n_poisoned} samples from class 1.")
        idx_1_poisoned = rng.integers(
            n0, size=n_poisoned
        )  # sample `n_poisoned` random indices from data0
        idx_1_true = rng.integers(
            n1, size=n1 - n_poisoned
        )  # sample `n-n_poisoned` random indices from data1
        dataY = np.concatenate((data0[idx_1_poisoned], data1[idx_1_true]), axis=0)
        rng.shuffle(dataY)
        Q = dataY
    else:
        raise ValueError("p_poisoning must be in [0,1].")
    return P, Q
