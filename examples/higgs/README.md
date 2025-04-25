# Higgs Experiments

This code reproduces the Higgs experiments of [Low-Rank Thinning](https://arxiv.org/pdf/2502.12063) (Section 6.2). To obtain the power and timing numbers in Figure 2 for all methods, please run:

```bash
./slurm/power_n16384_mixture.sh OUTPUT_PATH
```

> \[!TIP\]
> For quick debugging, you can run a similar experiment using `n=4096` points per sample, the first `d=4` features, and no data poisoning using the following command: `./slurm/power_n4096.sh OUTPUT_PATH`.

To generate the plot in Figure 2, please run:

```bash
python format_power.py -n 16384 -op OUTPUT_PATH
```

## Preparing an environment

Run the following commands to prepare a conda environment for the experiments:
```
conda create -n deepctt python=3.12
conda activate deepctt
pip install pandas tqdm matplotlib scipy tabulate
pip install git+https://github.com/microsoft/deepctt.git
```

## Downloading data

We follow the instructions from  [Liu et al. (2020)](https://github.com/fengliu90/DK-for-TST?tab=readme-ov-file#download-data), to download [HIGGS_TST.pckl](https://drive.google.com/open?id=1sHIIFCoHbauk6Mkb6e8a_tp1qnvuUOCc).
