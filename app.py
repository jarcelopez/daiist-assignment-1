"""
Gradio dashboard for Ames Housing Ridge models.

Loads training artifacts only — never retrains. Required views:
prediction vs actual (model + split selectors) and target distribution.
"""

from pathlib import Path

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ART = Path(__file__).resolve().parent / "artifacts"

SPLIT_CHOICES = [
    "train (2006–2008)",
    "val (2009)",
    "test (2010) — relevant for scoring",
]
SPLIT_KEYS = {
    "train (2006–2008)": "train",
    "val (2009)": "val",
    "test (2010) — relevant for scoring": "test",
}
MODEL_CHOICES = [
    "sklearn Ridge",
    "manual PyTorch",
    "standard PyTorch",
]


def _load_artifacts():
    X = {
        "train": np.load(ART / "X_train.npy"),
        "val": np.load(ART / "X_val.npy"),
        "test": np.load(ART / "X_test.npy"),
    }
    y_log = {
        "train": np.load(ART / "y_train.npy"),
        "val": np.load(ART / "y_val.npy"),
        "test": np.load(ART / "y_test.npy"),
    }

    sklearn_model = joblib.load(ART / "sklearn_model.joblib")

    manual = torch.load(ART / "pytorch_manual.pt", map_location="cpu", weights_only=False)
    w_manual = manual["w"].detach().cpu().numpy().ravel()
    b_manual = float(manual["b"])

    standard = torch.load(
        ART / "pytorch_standard.pt", map_location="cpu", weights_only=False
    )
    n_features = int(standard["n_features"])
    std_model = nn.Linear(n_features, 1, bias=True, dtype=torch.float64)
    std_model.load_state_dict(standard["state_dict"])
    std_model.eval()

    return X, y_log, sklearn_model, w_manual, b_manual, std_model


X_SPLITS, Y_LOG, SK_MODEL, W_MANUAL, B_MANUAL, STD_MODEL = _load_artifacts()


def predict_log(model_name: str, X: np.ndarray) -> np.ndarray:
    if model_name == "sklearn Ridge":
        return SK_MODEL.predict(X)
    if model_name == "manual PyTorch":
        return X @ W_MANUAL + B_MANUAL
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float64)
        return STD_MODEL(xt).numpy().ravel()


def plot_pred_vs_actual(model_name: str, split_label: str):
    split = SPLIT_KEYS[split_label]
    y_true = np.exp(Y_LOG[split])
    y_pred = np.exp(predict_log(model_name, X_SPLITS[split]))

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.45, s=18, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, label="perfect fit")
    ax.set_xlabel("Actual sale price ($)")
    ax.set_ylabel("Predicted sale price ($)")
    ax.set_title(f"{model_name} — {split_label}")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_target_distribution(split_label: str):
    split = SPLIT_KEYS[split_label]
    prices = np.exp(Y_LOG[split])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(prices, bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("Sale price ($)")
    ax.set_ylabel("Count")
    ax.set_title(f"Target distribution — {split_label}")
    fig.tight_layout()
    return fig


NOTE = (
    "Models are loaded from `artifacts/` only; nothing is retrained here. "
    "Use **test (2010)** for the score that matters; train and val are for "
    "inspection. Predictions are dollars after undoing log(SalePrice)."
)

with gr.Blocks(title="Ames Housing — model comparison") as demo:
    gr.Markdown("# Ames Housing dashboard")
    gr.Markdown(NOTE)

    with gr.Tab("Prediction vs actual"):
        with gr.Row():
            model_dd = gr.Dropdown(
                MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model"
            )
            split_dd = gr.Dropdown(
                SPLIT_CHOICES, value=SPLIT_CHOICES[2], label="Split"
            )
        pred_plot = gr.Plot(label="Prediction vs actual")
        pred_btn = gr.Button("Update plot")
        pred_btn.click(
            plot_pred_vs_actual, inputs=[model_dd, split_dd], outputs=pred_plot
        )
        demo.load(
            plot_pred_vs_actual, inputs=[model_dd, split_dd], outputs=pred_plot
        )

    with gr.Tab("Target distribution"):
        dist_split = gr.Dropdown(
            SPLIT_CHOICES, value=SPLIT_CHOICES[2], label="Split"
        )
        dist_plot = gr.Plot(label="SalePrice distribution")
        dist_btn = gr.Button("Update plot")
        dist_btn.click(
            plot_target_distribution, inputs=dist_split, outputs=dist_plot
        )
        demo.load(
            plot_target_distribution, inputs=dist_split, outputs=dist_plot
        )


if __name__ == "__main__":
    demo.launch()
