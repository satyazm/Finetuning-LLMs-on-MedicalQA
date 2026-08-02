import json

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "baseline": "#2a78d6",
    "lora": "#eb6834",
    "qlora": "#1baf7a",
}
LABELS = {
    "baseline": "Baseline",
    "lora": "LoRA",
    "qlora": "QLoRA",
}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c8c0")
    ax.spines["bottom"].set_color("#c9c8c0")
    ax.tick_params(colors="#52514e")
    ax.yaxis.grid(True, color="#e5e4dd", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_loss_curve(history_paths, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    epochs = []
    for name, path in history_paths.items():
        with open(path) as f:
            history = json.load(f)
        epochs = [h["epoch"] for h in history]
        color = COLORS[name]
        ax.plot(epochs, [h["training_loss"] for h in history], color=color, linewidth=2,
                 linestyle="-", marker="o", markersize=5, label=f"{LABELS[name]} (train)")
        ax.plot(epochs, [h["validation_loss"] for h in history], color=color, linewidth=2,
                 linestyle="--", marker="o", markersize=5, label=f"{LABELS[name]} (val)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    if epochs:
        ax.set_xticks(epochs)
    _style_axes(ax)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _grouped_bar_chart(df, columns, xticklabels, output_path, title, ylabel, note=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    n_groups = len(columns)
    n_models = len(df)
    width = 0.8 / n_models
    x = range(n_groups)

    for i, model in enumerate(df.index):
        values = df.loc[model, columns].astype(float).tolist()
        offsets = [xi - 0.4 + width * (i + 0.5) for xi in x]
        ax.bar(offsets, values, width=width, color=COLORS[model], label=LABELS[model])

    ax.set_xticks(list(x))
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=n_models)

    fig.tight_layout()
    bottom = 0.28 if note else 0.2
    fig.subplots_adjust(bottom=bottom)
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=8, color="#7a7971", wrap=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_rouge_comparison(comparison_csv_path, output_path):
    df = pd.read_csv(comparison_csv_path, index_col="model")
    _grouped_bar_chart(df, ["rouge1", "rouge2", "rougeL"], ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
                        output_path, "ROUGE comparison", "Score")


def plot_bertscore_comparison(comparison_csv_path, output_path):
    df = pd.read_csv(comparison_csv_path, index_col="model").dropna(subset=["bertscore_f1"])
    _grouped_bar_chart(df, ["bertscore_f1"], ["BERTScore F1"], output_path, "BERTScore comparison", "Score")


def plot_latency_comparison(comparison_csv_path, output_path):
    df = pd.read_csv(comparison_csv_path, index_col="model")
    _grouped_bar_chart(df, ["mean_latency_sec"], ["Mean latency"], output_path,
                        "Generation latency", "Seconds")


def plot_memory_comparison(comparison_csv_path, output_path):
    df = pd.read_csv(comparison_csv_path, index_col="model")
    note = ("All three generate from an unquantized fp16 base model at inference time, so\n"
            "usage should be similar; QLoRA's higher reading likely reflects a one-time\n"
            "loading-time cast rather than a true generation-time difference.")
    _grouped_bar_chart(df, ["gpu_memory_gb"], ["Inference memory"], output_path,
                        "Peak inference GPU memory", "GB", note=note)
