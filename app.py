
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from huggingface_hub import hf_hub_download

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -------------------------------
# 1. APP CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="FedHealthOps",
    page_icon="🏥",
    layout="wide"
)

MODEL_REPO = "affu678/fedhealthops-diabetes"
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

TARGET = "Outcome"
torch.manual_seed(42)
np.random.seed(42)


# -------------------------------
# 2. PYTORCH MODEL
# -------------------------------
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


# -------------------------------
# 3. LOAD SAVED MODEL AND SCALER
# -------------------------------
@st.cache_resource
def load_saved_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="global_model.pth"
    )

    scaler_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="scaler.joblib"
    )

    model = DiabetesNet()

    try:
        state = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler


@st.cache_data
def load_saved_metrics():
    try:
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="metrics.json"
        )

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return {}


def get_metric(metrics, names):
    for name in names:
        try:
            value = float(metrics[name])
            if value <= 1:
                value *= 100
            return f"{value:.1f}%"
        except (KeyError, TypeError, ValueError):
            continue

    return "Not logged"


# -------------------------------
# 4. LOAD TRAINING DATA
# -------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_URL)

    missing = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    X = df[FEATURES].astype(float).copy()
    y = df[TARGET].astype(int).copy()

    # Replace impossible zero values with column medians
    for col in [
        "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI"
    ]:
        X.loc[X[col] == 0, col] = np.nan
        X[col] = X[col].fillna(X[col].median())

    return X, y


# -------------------------------
# 5. FEDERATED AVERAGING
# -------------------------------
def average_weights(client_states, client_sizes):
    total = sum(client_sizes)
    averaged = {}

    for key in client_states[0]:
        averaged[key] = sum(
            client_states[i][key] *
            (client_sizes[i] / total)
            for i in range(len(client_states))
        )

    return averaged


def copy_weights(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


# -------------------------------
# 6. FEDERATED TRAINING
# -------------------------------
def train_federated(rounds, epochs, learning_rate):
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Split training samples among 3 simulated clients
    splits = np.array_split(
        np.random.permutation(len(X_train)), 3
    )

    clients = []

    for indices in splits:
        client_x = torch.tensor(
            X_train[indices], dtype=torch.float32
        )

        client_y = torch.tensor(
            y_train.iloc[indices].to_numpy().reshape(-1, 1),
            dtype=torch.float32
        )

        clients.append((client_x, client_y))

    global_model = DiabetesNet()
    results = []

    progress = st.progress(0)

    for round_no in range(rounds):
        client_states = []
        client_sizes = []
        losses = []

        for client_x, client_y in clients:
            local_model = DiabetesNet()
            local_model.load_state_dict(
                global_model.state_dict()
            )

            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=learning_rate
            )

            loss_function = nn.BCELoss()
            local_model.train()

            for _ in range(epochs):
                optimizer.zero_grad()

                predictions = local_model(client_x)
                loss = loss_function(predictions, client_y)

                loss.backward()
                optimizer.step()

            client_states.append(copy_weights(local_model))
            client_sizes.append(len(client_x))
            losses.append(loss.item())

        # Aggregate client models using FedAvg
        global_model.load_state_dict(
            average_weights(client_states, client_sizes)
        )

        global_model.eval()

        with torch.no_grad():
            probabilities = global_model(
                torch.tensor(X_test, dtype=torch.float32)
            ).numpy().ravel()

        predicted = (probabilities >= 0.5).astype(int)

        results.append({
            "Round": round_no + 1,
            "Accuracy": accuracy_score(y_test, predicted),
            "Precision": precision_score(
                y_test, predicted, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predicted, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predicted, zero_division=0
            ),
            "Mean Local Loss": float(np.mean(losses))
        })

        progress.progress(
            (round_no + 1) / rounds,
            text=f"Completed round {round_no + 1}/{rounds}"
        )

    return pd.DataFrame(results)


# -------------------------------
# 7. SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("🏥 FedHealthOps")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Federated Training",
        "Demo Prediction",
        "Model & MLOps"
    ]
)

st.title("🏥 FedHealthOps")
st.caption("Privacy-Preserving Federated Learning Demo")

st.warning(
    "Educational demonstration only. The three clients are simulated "
    "inside one application. This model is not clinically validated. "
    "Do not use predictions for medical decisions."
)


# -------------------------------
# 8. OVERVIEW
# -------------------------------
if page == "Overview":

    st.header("Saved Model Performance")

    metrics = load_saved_metrics()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        get_metric(metrics, ["accuracy", "test_accuracy", "acc"])
    )

    col2.metric(
        "Precision",
        get_metric(metrics, ["precision", "test_precision"])
    )

    col3.metric(
        "Recall",
        get_metric(metrics, ["recall", "test_recall"])
    )

    col4.metric(
        "F1 Score",
        get_metric(metrics, ["f1", "f1_score", "f1-score"])
    )

    st.subheader("Project Workflow")

    st.write("1. Divide the training data among three simulated clients.")
    st.write("2. Train a local neural network at each client.")
    st.write("3. Aggregate model weights using FedAvg.")
    st.write("4. Evaluate the model after each round.")
    st.write("5. Use the saved model for an example prediction.")

    st.link_button(
        "Open Hugging Face Model",
        f"https://huggingface.co/{MODEL_REPO}"
    )

    if st.button("Refresh saved metrics"):
        load_saved_metrics.clear()
        st.rerun()


# -------------------------------
# 9. FEDERATED TRAINING PAGE
# -------------------------------
elif page == "Federated Training":

    st.header("Federated Training Simulation")

    st.info(
        "This runs a new training simulation on a public dataset. "
        "It does not overwrite your saved model or automatically "
        "create an MLflow run."
    )

    col1, col2, col3 = st.columns(3)

    rounds = col1.slider("Federated Rounds", 1, 10, 5)
    epochs = col2.slider("Local Epochs", 1, 10, 3)

    learning_rate = col3.slider(
        "Learning Rate",
        0.001,
        0.1,
        0.01,
        step=0.001
    )

    if st.button("▶ Start Federated Training", type="primary"):

        try:
            with st.spinner("Training simulated clients..."):
                results = train_federated(
                    rounds, epochs, learning_rate
                )

            st.success("Federated training completed!")

            latest = results.iloc[-1]

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", f"{latest['Accuracy']:.1%}")
            c2.metric("Precision", f"{latest['Precision']:.1%}")
            c3.metric("Recall", f"{latest['Recall']:.1%}")
            c4.metric("F1 Score", f"{latest['F1 Score']:.1%}")

            st.subheader("Training Metrics Graph")

            chart = results.set_index("Round")[
                ["Accuracy", "Precision", "Recall", "F1 Score"]
            ]

            st.line_chart(chart)

            st.subheader("Round-by-Round Results")

            st.dataframe(
                results.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1 Score": "{:.2%}",
                    "Mean Local Loss": "{:.4f}"
                }),
                use_container_width=True
            )

            st.download_button(
                "Download Training Results CSV",
                results.to_csv(index=False).encode("utf-8"),
                "fedhealthops_results.csv",
                "text/csv"
            )

        except Exception as error:
            st.error(
                f"Training failed: {type(error).__name__}: {error}"
            )


# -------------------------------
# 10. SAVED MODEL PREDICTION
# -------------------------------
elif page == "Demo Prediction":

    st.header("Diabetes Model Demo")

    st.info(
        "Enter synthetic example values only. The output is not "
        "a diagnosis or a clinically validated risk estimate."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        pregnancies = col1.number_input(
            "Pregnancies", min_value=0, max_value=30, value=2
        )
        glucose = col1.number_input(
            "Glucose", min_value=0.0, value=120.0
        )
        bp = col1.number_input(
            "Blood Pressure", min_value=0.0, value=70.0
        )
        skin = col1.number_input(
            "Skin Thickness", min_value=0.0, value=20.0
        )

        insulin = col2.number_input(
            "Insulin", min_value=0.0, value=80.0
        )
        bmi = col2.number_input(
            "BMI", min_value=0.0, value=28.5
        )
        pedigree = col2.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=0.35
        )
        age = col2.number_input(
            "Age", min_value=1, max_value=120, value=33
        )

        threshold = st.slider(
            "Classification Threshold", 0.1, 0.9, 0.5, 0.05
        )

        submitted = st.form_submit_button(
            "Run Prediction", type="primary"
        )

    if submitted:

        try:
            model, scaler = load_saved_model()

            values = np.array([[
                pregnancies, glucose, bp, skin,
                insulin, bmi, pedigree, age
            ]], dtype=float)

            scaled_values = scaler.transform(values)

            with torch.no_grad():
                score = float(
                    model(
                        torch.tensor(
                            scaled_values, dtype=torch.float32
                        )
                    ).item()
                )

            st.metric("Model Output Score", f"{score:.1%}")

            if score >= threshold:
                st.warning("The model predicts the positive class.")
            else:
                st.info("The model predicts the negative class.")

            st.caption(
                "This is an educational model output, not medical advice."
            )

        except Exception as error:
            st.error(
                f"Prediction failed: {type(error).__name__}: {error}"
            )

            st.write(
                "Check that global_model.pth and scaler.joblib "
                f"exist in the Hugging Face repository: {MODEL_REPO}"
            )


# -------------------------------
# 11. MODEL AND MLOPS
# -------------------------------
elif page == "Model & MLOps":

    st.header("Model Information")

    st.write("**Framework:** PyTorch")
    st.write("**Algorithm:** Federated Averaging (FedAvg)")
    st.write("**Neural Network:** 8 input features, 16 hidden units, 1 output")
    st.write("**Clients:** Three simulated data partitions")
    st.write("**Experiment Tracking:** MLflow run from the earlier notebook")
    st.write("**Model Storage:** Hugging Face")

    st.link_button(
        "View Model Repository",
        f"https://huggingface.co/{MODEL_REPO}"
    )

    st.caption(
        "Training started here is an in-memory simulation. It does not "
        "automatically save new model files to Hugging Face or log new "
        "experiments to MLflow."
    )
