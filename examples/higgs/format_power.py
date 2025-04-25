"""Script to format the time-power trade-off curves for the Higgs experiment.

Example usage:
```bash
python format_power.py -n 4096 -op /path/to/output/dir
```
"""

import numpy as np
import pandas as pd
from glob import glob
import os
import matplotlib.pyplot as plt
from util_experiments import get_base_parser

# Enable LaTeX for text rendering
import matplotlib as mpl
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amssymb} \usepackage{amsmath}"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

parser = get_base_parser()
args = parser.parse_args()
n = args.n
output_path = args.output_path

results_dir = os.path.join(
    output_path,
    "results",
)
results_path = os.path.join(results_dir, f"*-n{n}-g*-kk*.csv")
results_files = glob(results_path)
print(f"Found {len(results_files)} matches for {results_path}")
df_separate = pd.concat([pd.read_csv(f) for f in results_files])
df = pd.concat([pd.read_csv(f) for f in results_files])
df = df[df["method"] != "ctt0"]
df = df.fillna(np.inf)


def perc_mean_pm_stderr(x: np.ndarray) -> str:
    """Format the mean percentage and standard error of a numpy array as a string."""
    return f"{np.mean(x) * 100:.2f} ± {np.std(x, ddof=1) / np.sqrt(len(x)) * 100:.2f}"


def abs_mean_pm_stderr(x: np.ndarray) -> str:
    """Format the mean and standard error of a numpy array as a string."""
    return f"{np.mean(x):.4f} ± {np.std(x, ddof=1) / np.sqrt(len(x)):.4f}"


df_by_method_g = df.groupby(["method", "g"]).agg(
    {"H": abs_mean_pm_stderr, "time": abs_mean_pm_stderr}
)
# NOTE: subsampling with g=0 is identical to wb_block with g=0
if ("subsampling", 0.0) not in df_by_method_g.index:
    # Create a new index level for subsampling with g=0.0
    new_index = pd.MultiIndex.from_tuples([("subsampling", 0.0)], names=["method", "g"])
    # Add the new row
    df_by_method_g = pd.concat([df_by_method_g, pd.DataFrame(index=new_index)])
    # Now you can assign the values
    df_by_method_g.loc[("subsampling", 0.0)] = df_by_method_g.loc[("wb_block", 0.0)]
# sort the index
df_by_method_g = df_by_method_g.sort_index()


# add number of entries
def get_num_entries(method: str, n: int, g: int) -> int:
    """Get the number of kernel evaluations (entries) for a given method, number of samples, and thinning parameter.

    Args:
        method (str): the method to use
        n (int): the number of samples
        g (int): the thinning parameter

    Returns:
        int: the number of kernel evaluations

    """
    if g == np.inf:
        entries = n**2
    else:
        if method == "wb_block":
            entries = n**2 / 2**g
        # NOTE: change the method name to wb_standard for backward compatibility with old results
        # elif method == "wb_standard":
        elif method == "subsampling":
            entries = (n / 2**g) ** 2
        elif method.startswith("ctt"):
            num_bins = 16  # see ctt.py
            entries = 4**g * n * num_bins
        else:
            raise ValueError(f"Unknown method: {method}")
    return int(entries)


df_by_method_g["entries"] = [
    get_num_entries(method, n, g) for method, g in df_by_method_g.index
]
df_by_method_g["log2(entries)"] = np.round(np.log2(df_by_method_g["entries"])).astype(
    int
)
print(df_by_method_g.to_markdown())

fig, ax = plt.subplots(figsize=(3.25, 2.5))
labels = {
    "wb_block": r"W-Block",
    "subsampling": r"Subsampling",
    "ctt": r"CTT",
}
colors = {
    "wb_block": "tab:red",
    "subsampling": "tab:green",
    "ctt": "tab:blue",
}
linestyle = {
    "wb_block": "--",
    "subsampling": "dotted",
    "ctt": "-",
}
smallest_log2_entries = df_by_method_g["log2(entries)"].min()
largest_log2_entries = df_by_method_g["log2(entries)"].max()
for method in df_by_method_g.index.levels[0]:
    df_method = df_by_method_g.loc[method]
    x = df_method["time"].apply(lambda x: float(x.split(" ± ")[0]))
    xerr = df_method["time"].apply(lambda x: float(x.split(" ± ")[1]))
    y = df_method["H"].apply(lambda x: float(x.split(" ± ")[0]))
    yerr = df_method["H"].apply(lambda x: float(x.split(" ± ")[1]))
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o-",
        label=labels[method],
        color=colors[method],
        linestyle=linestyle[method],
        markersize=0,
    )
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.3, color=colors[method])
    marker_sizes = df_method["log2(entries)"].apply(
        lambda x: 5
        + 50
        * (x - smallest_log2_entries)
        / (largest_log2_entries - smallest_log2_entries)
    )
    ax.scatter(x, y, color=colors[method], s=marker_sizes)
    for g, (H, time, entries, log2_entries) in df_method.iterrows():
        if method.startswith("ctt"):
            if g > 2:
                text = rf"g={int(g)}"
            else:
                text = ""
            deltay = 0
            deltax = 1.5
        elif method == "wb_block":
            if g <= 2:  # only label the first few points from the right
                text = "$B=2^{{{}}}$".format(int((n.bit_length() - 1) - g))
            else:
                text = ""
            deltay = 0.08
            deltax = -1.5
        elif method == "subsampling":
            if g == 0:
                text = "$n$"
            elif int(g) == g and g <= 1:
                text = f"$\\frac{{n}}{{{int(2**g)}}}$"
            else:
                text = ""
            deltay = -0.1
            deltax = 0.3
        else:
            text = ""
            deltay = deltax = 0

        ax.text(
            x=float(time.split(" ± ")[0]) + deltax,
            y=float(H.split(" ± ")[0]) + deltay,
            s=text,
            color=colors[method],
            fontsize=8,
        )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_position(("outward", 1))
ax.spines["left"].set_position(("outward", 1))

ax.grid(True)
ax.grid(
    which="both", axis="both", color="gray", linestyle="--", linewidth=0.1, alpha=0.25
)

alpha = 0.05
ax.axhline(
    alpha,
    color="orange",
    linestyle="--",
    label=rf"Level {alpha}",
    linewidth=1,
    alpha=0.75,
)
ax.set_xlabel(r"Time per test (ms)", fontsize=10)
max_time = max(float(time.split(" ± ")[0]) for time in df_by_method_g["time"])
ax.set_xlim(left=0, right=10 * (max_time // 10) + 10)
ax.set_ylabel(r"Power (1 - Type II Error)", fontsize=10)
ax.set_ylim(0, 1)
ax.tick_params(axis="both", which="major", labelsize=8)

handles, labels = ax.get_legend_handles_labels()
order = [1, 3, 2, 0]
ax.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    fontsize=8,
    loc="right",
    bbox_to_anchor=(1, 0.35),
)

plt.tight_layout()
save_fig_dir = os.path.join(output_path, "figures")
os.makedirs(save_fig_dir, exist_ok=True)
save_fig_path = os.path.join(save_fig_dir, "power.png")
plt.savefig(save_fig_path)
print(f"Saved figure to {save_fig_path}")
save_fig_path = os.path.join(save_fig_dir, "power.pdf")
plt.savefig(save_fig_path)
print(f"Saved figure to {save_fig_path}")
