import json, os, time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np, scipy.stats as st

def _load(path):
    with open(path) as fp:
        return json.load(fp)

def numeric_annotation(ax):
    for line in ax.get_lines():
        x, y = line.get_xdata()[-1], line.get_ydata()[-1]
        ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(3, 0),
                    textcoords="offset points", fontsize=6)

# ---------------------------------------------------
def plot_training(log, fig_path):
    sns.set_theme(style="darkgrid"); plt.figure(figsize=(6,3))
    ax = sns.lineplot(x=range(len(log["train_loss"])), y=log["train_loss"], label="loss")
    sns.lineplot(x=range(len(log["kl"])), y=log["kl"], label="KL")
    numeric_annotation(ax)
    ax.set_xlabel("step"); ax.set_ylabel("value")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight"); plt.close()

# ---------------------------------------------------
def evaluate(json_log, description):
    log = _load(json_log)
    log_path = Path(json_log)
    image_dir = Path('.research/iteration4/images')
    image_dir.mkdir(parents=True, exist_ok=True)
    fig_path = image_dir / log_path.with_suffix(".pdf").name
    plot_training(log, fig_path)

    # ------------ numerical summary ---------------
    final_loss = np.mean(log["train_loss"][-10:])
    final_kl   = np.mean(log["kl"][-10:])
    beta_mean  = np.mean(log["beta"][-10:])
    results = {"final_loss": final_loss,
               "final_kl": final_kl,
               "beta_mean": beta_mean,
               "steps": len(log["train_loss"]),
               "figure": str(fig_path)}
    res_path = log_path.with_suffix(".results.json")
    with open(res_path, "w") as fp: json.dump(results, fp, indent=2)

    # -------------- STDOUT ------------------------
    print("\nEXPERIMENT DESCRIPTION\n----------------------")
    print(description)
    print("\nNUMERICAL RESULTS\n-----------------")
    print(json.dumps(results, indent=2))
    print("\nGENERATED FIGURES\n-----------------")
    print(fig_path.name)
    return results
