import os, json, copy, time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from huggingface_hub import hf_hub_download

# Educational demo only. The three clients are simulated in one app/runtime.
MODEL_REPO = os.getenv("MODEL_REPO", "affu678/fedhealthops-diabetes")
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"
DEVICE = torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)

class DiabetesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def load_artifact(filename):
    return hf_hub_download(repo_id=MODEL_REPO, filename=filename)

def load_model_and_scaler():
    model = DiabetesNet().to(DEVICE)
    model_path = load_artifact("global_model.pth")
    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=DEVICE)
    # Support either a plain state_dict or a checkpoint wrapper.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    scaler_path = load_artifact("scaler.joblib")
    import joblib
    scaler = joblib.load(scaler_path)
    return model, scaler

def load_metrics():
    try:
        p = load_artifact("metrics.json")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def metric_value(metrics, key):
    aliases = {
        "accuracy": ["accuracy", "test_accuracy", "acc"],
        "precision": ["precision", "test_precision"],
        "recall": ["recall", "test_recall"],
        "f1": ["f1", "f1_score", "f1-score"]
    }
    for name in aliases.get(key, [key]):
        if name in metrics:
            try:
                return f"{float(metrics[name]) * (100 if float(metrics[name]) <= 1 else 1):.1f}%"
            except Exception:
                pass
    return "Not logged"

def overview():
    metrics = load_metrics()
    cards = (
        f"### Accuracy\n{metric_value(metrics, 'accuracy')}\n\n"
        f"### Precision\n{metric_value(metrics, 'precision')}\n\n"
        f"### Recall\n{metric_value(metrics, 'recall')}\n\n"
        f"### F1 score\n{metric_value(metrics, 'f1')}"
    )
    return cards

def predict(pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age, threshold):
    values = [pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]
    if any(v is None for v in values):
        return "Please enter all eight sample features."
    try:
        model, scaler = load_model_and_scaler()
        x = np.asarray(values, dtype=float).reshape(1, -1)
        x_scaled = scaler.transform(x)
        with torch.no_grad():
            prob = float(model(torch.tensor(x_scaled, dtype=torch.float32)).item())
        label = "Positive class predicted" if prob >= threshold else "Negative class predicted"
        return (
            f"## {label}\n"
            f"**Model score:** {prob:.1%}\n\n"
            f"Threshold: {threshold:.2f}\n\n"
            "Educational demonstration only. This is not a diagnosis or a clinically validated risk estimate."
        )
    except Exception as e:
        return (
            "Could not load the model artifacts or run inference.\n\n"
            f"Details: `{type(e).__name__}: {e}`\n\n"
            "Check that global_model.pth and scaler.joblib exist in the Hugging Face model repository."
        )

def load_dataset():
    df = pd.read_csv(DATA_URL)
    # Dataset is expected to have the Pima-style columns used by the notebook.
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset columns missing: {missing}")
    X = df[FEATURES].astype(float).copy()
    y = df[TARGET].astype(int).copy()
    # Replace impossible zero values in selected clinical fields with medians for this demo.
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        X.loc[X[col] == 0, col] = np.nan
        X[col] = X[col].fillna(X[col].median())
    return X, y

def make_model():
    return DiabetesNet()

def state_to_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def fedavg(states, weights):
    total = float(sum(weights))
    result = {}
    for key in states[0]:
        result[key] = sum(states[i][key] * (weights[i] / total) for i in range(len(states)))
    return result

def run_federated_training(rounds, epochs, lr):
    try:
        X, y = load_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Split training rows into three simulated hospital clients.
        splits = np.array_split(np.random.permutation(len(X_train_scaled)), 3)
        clients = []
        for ids in splits:
            clients.append((
                torch.tensor(X_train_scaled[ids], dtype=torch.float32),
                torch.tensor(y_train.to_numpy()[ids].reshape(-1, 1), dtype=torch.float32)
            ))

        global_model = make_model()
        rows = []
        for round_idx in range(int(rounds)):
            client_states = []
            client_sizes = []
            for client_x, client_y in clients:
                local = make_model()
                local.load_state_dict(global_model.state_dict())
                optimizer = torch.optim.SGD(local.parameters(), lr=float(lr))
                loss_fn = nn.BCELoss()
                local.train()
                for _ in range(int(epochs)):
                    optimizer.zero_grad()
                    pred = local(client_x)
                    loss = loss_fn(pred, client_y)
                    loss.backward()
                    optimizer.step()
                client_states.append(state_to_cpu(local))
                client_sizes.append(len(client_x))
            global_model.load_state_dict(fedavg(client_states, client_sizes))
            global_model.eval()
            with torch.no_grad():
                pred_prob = global_model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().ravel()
            pred = (pred_prob >= 0.5).astype(int)
            rows.append({
                "Round": round_idx + 1,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "Mean local loss": float(loss.item())
            })

        results = pd.DataFrame(rows)
        # Plot actual evaluation metrics measured after each round.
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in ["Accuracy", "Precision", "Recall", "F1"]:
            ax.plot(results["Round"], results[col], marker="o", label=col)
        ax.set_xlabel("Federated round")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title("Evaluation metrics after each FedAvg round")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()

        last = results.iloc[-1]
        summary = (
            f"### Latest federated evaluation\n"
            f"- Accuracy: **{last['Accuracy']:.1%}**\n"
            f"- Precision: **{last['Precision']:.1%}**\n"
            f"- Recall: **{last['Recall']:.1%}**\n"
            f"- F1 score: **{last['F1']:.1%}**\n\n"
            f"Completed **{int(rounds)} rounds** with **3 simulated clients**. "
            "These clients run in one app environment; this is not a deployment across real hospitals."
        )
        return summary, fig, results, "Training completed."
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.02, 0.5, f"Training failed: {type(e).__name__}: {e}", wrap=True)
        ax.axis("off")
        return f"Training failed: `{type(e).__name__}: {e}`", fig, pd.DataFrame(), "Training failed."

with gr.Blocks(title="FedHealthOps | Federated Health Intelligence", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🏥 FedHealthOps\n"
        "### Privacy-aware federated learning research workspace\n"
        "Train three **simulated** hospital clients, aggregate model weights with FedAvg, "
        "inspect evaluation charts, and test the saved model with example data."
    )
    with gr.Tab("Overview"):
        gr.Markdown("Metrics loaded from your Hugging Face model artifacts. Values appear as “Not logged” if a metric key is absent.")
        refresh_btn = gr.Button("Refresh saved model metrics")
        metrics_md = gr.Markdown(overview())
        refresh_btn.click(fn=overview, outputs=metrics_md)
        gr.Markdown(
            "> **Research demo:** No real hospitals are connected. Do not enter patient-identifiable data. "
            "Predictions are educational and are not medical advice."
        )
    with gr.Tab("Federated training"):
        gr.Markdown(
            "This tab runs a fresh demonstration of federated averaging on a public diabetes dataset. "
            "Each client trains locally on its partition; only model weights are averaged in this simulation."
        )
        with gr.Row():
            rounds_in = gr.Slider(1, 10, value=5, step=1, label="Federated rounds")
            epochs_in = gr.Slider(1, 10, value=3, step=1, label="Local epochs per round")
            lr_in = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Learning rate")
        train_btn = gr.Button("▶ Run federated training", variant="primary")
        train_status = gr.Markdown("Ready to train.")
        train_summary = gr.Markdown()
        train_plot = gr.Plot(label="Live round metrics")
        train_table = gr.Dataframe(label="Round-by-round evaluation", interactive=False)
        train_btn.click(
            fn=run_federated_training,
            inputs=[rounds_in, epochs_in, lr_in],
            outputs=[train_summary, train_plot, train_table, train_status]
        )
    with gr.Tab("Demo prediction"):
        gr.Markdown(
            "Enter **synthetic/example values only**. This uses the saved PyTorch model and scaler from "
            f"[{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})."
        )
        with gr.Row():
            with gr.Column():
                preg = gr.Number(value=2, label="Pregnancies")
                glucose = gr.Number(value=120, label="Glucose")
                bp = gr.Number(value=70, label="Blood pressure")
                skin = gr.Number(value=20, label="Skin thickness")
            with gr.Column():
                insulin = gr.Number(value=80, label="Insulin")
                bmi = gr.Number(value=28.5, label="BMI")
                pedigree = gr.Number(value=0.35, label="Diabetes pedigree function")
                age = gr.Number(value=33, label="Age")
        threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Demo classification threshold")
        predict_btn = gr.Button("Run prediction with saved model", variant="primary")
        prediction = gr.Markdown("Prediction output will appear here.")
        predict_btn.click(
            fn=predict,
            inputs=[preg, glucose, bp, skin, insulin, bmi, pedigree, age, threshold],
            outputs=prediction
        )
        gr.Markdown(
            "**Important:** This public dataset and model are for coursework demonstrations only. "
            "The output is not a clinical risk assessment or diagnosis."
        )
    with gr.Tab("Model & MLOps"):
        gr.Markdown(
            f"- **Model artifacts:** [Hugging Face model repository](https://huggingface.co/{MODEL_REPO})\n"
            "- **Framework:** PyTorch\n"
            "- **Aggregation:** Federated Averaging (FedAvg)\n"
            "- **Tracking:** MLflow (the original notebook run remains separate from this app's new training sessions)\n"
            "- **Data:** Public Pima-style diabetes dataset\n\n"
            "Training performed from this app is an in-memory demonstration. It does not automatically overwrite "
            "the hosted model artifacts or log new runs to MLflow."
        )

if __name__ == "__main__":
    demo.launch()
