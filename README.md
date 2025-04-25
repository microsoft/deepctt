# Deep Compress-Then-Test

**Deep Compress-Then-Test** (Deep CTT) accelerates deep kernel two-sample testing using high-fidelity compression.

For a detailed description of the **Deep CTT** algorithm and its power guarantees, see [Low-Rank Thinning](https://arxiv.org/pdf/2502.12063).

```bibtex
@article{carrell2025low,
  title={Low-Rank Thinning},
  author={Carrell, Annabelle Michael and Gong, Albert and Shetty, Abhishek and Dwivedi, Raaz and Mackey, Lester},
  journal={arXiv preprint arXiv:2502.12063},
  year={2025}
}
```

## Getting Started

To install the `deepctt` package, use the following pip command:

```bash
pip install git+https://github.com/microsoft/deepctt.git
```

To test whether two samples, X and Y, are drawn from a common distribution, please follow these steps:

```python
from deepctt import ctt
from deepctt.utils import train_deep_kernel
import torch

# Assumes the samples X and Y are numpy arrays of shape (n1,d) and (n2,d), respectively
n1, _ = X.shape
n2, _ = Y.shape
X_train, Y_train = X[:n1//2], Y[:n2//2]
X_test, Y_test = X[n1//2:], Y[n2//2:]

# Fit the deep kernel
model, sigma0, sigma, ep = train_deep_kernel(
  X_train, Y_train, N_epoch, device, dtype, input_dim, learning_rate=5e-5, hidden_dim=20, embedding_dim=20
)
rejects, threshold_values, statistic_values = ctt(
  torch.cat((model(X_test), X_test), dim=1),
  torch.cat((model(Y_test), Y_test), dim=1),
  g=0,  # oversampling parameter
  B=100,  # number of permutations
  alpha=0.05,  # nominal level
  sigma0=sigma0,
  sigma=sigma,
  ep=ep,
  d_embd=embedding_dim,
)
```

For an example usage, see our [Higgs experiments](./examples/higgs/README.md).

This package has been tested with the following operating system, Python, and PyTorch combintations:
- Ubuntu 20.04, Python 3.12.9, Torch 2.6.0
- Ubuntu 20.04, Python 3.12.9, Torch 2.4.0

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
