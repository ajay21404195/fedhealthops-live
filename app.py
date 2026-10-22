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

# ---------------------------------
# 1. APP CONFIGURATION
# ---------------------------------
st.set_page_config(
    page_title="FedHealthOps",
    page_icon="🏥",
    layout="wide"
)

MODEL_REPO = "affu678/fedhealthops-diabetes"
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET = "Outcome"

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------
# 2. PYTORCH MODEL
# ---------------------------------
class DiabetesNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------
# 3. LOAD SAVED MODEL
# ---------------------------------
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
        state = torch.load(
            model_path,
            map_location="cpu"
        )

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


# ---------------------------------
# 4. DATA PREPARATION
# ---------------------------------
def prepare_dataset(df):
    missing = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    X = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    ).copy()

    y = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    valid_rows = y.notna() & y.isin([0, 1])

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int).copy()

    for col in [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]:
        X.loc[X[col] == 0, col] = np.nan
        X[col] = X[col].fillna(X[col].median())

    valid_features = (
        X.notna().all(axis=1)
        & np.isfinite(X).all(axis=1)
    )

    X = X.loc[valid_features]
    y = y.loc[valid_features]

    if len(X) < 20:
        raise ValueError(
            "Each hospital dataset must have at least 20 valid rows."
        )

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError(
            "Outcome must contain both 0 and 1, "
            "with at least two rows for each class."
        )

    return (
        X.reset_index(drop=True),
        y.reset_index(drop=True)
    )


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_URL)
    return prepare_dataset(df)


# ---------------------------------
# 5. FEDERATED AVERAGING
# ---------------------------------
def copy_weights(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def average_weights(client_states, client_sizes):
    total = sum(client_sizes)

    if total == 0:
        raise ValueError("No local training data is available.")

    averaged = {}

    for key in client_states[0]:
        averaged[key] = sum(
            client_states[i][key] * (
                client_sizes[i] / total
            )
            for i in range(len(client_states))
        )

    return averaged


# ---------------------------------
# 6. FEDERATED TRAINING
# ---------------------------------
def train_federated(
    rounds,
    epochs,
    learning_rate,
    hospital_data,
    chart_placeholder=None,
    table_placeholder=None
):
    if not hospital_data:
        raise ValueError("Select at least one hospital.")

    # Load public demo data for hospitals without uploaded CSV files.
    fallback_X, fallback_y = load_dataset()

    fallback_df = fallback_X.copy()
    fallback_df[TARGET] = fallback_y.to_numpy()

    fallback_parts = np.array_split(
        np.random.RandomState(42).permutation(
            len(fallback_df)
        ),
        len(hospital_data)
    )

    prepared_clients = []
    all_X = []
    all_y = []

    # Prepare a separate dataset for each simulated hospital.
    row_offset = 0

    for index, (hospital_name, hospital_df) in enumerate(
        hospital_data.items()
    ):
        if hospital_df is None:
            hospital_df = fallback_df.iloc[
                fallback_parts[index]
            ].copy()

        X_local, y_local = prepare_dataset(hospital_df)

        # Assign unique row indexes across hospitals.
        unique_index = range(
            row_offset,
            row_offset + len(X_local)
        )

        X_local.index = unique_index
        y_local.index = unique_index

        row_offset += len(X_local)

        prepared_clients.append(
            (hospital_name, X_local, y_local)
        )

        all_X.append(X_local)
        all_y.append(y_local)

    # Combine rows for splitting and evaluation.
    X = pd.concat(all_X)
    y = pd.concat(all_y)

    if y.value_counts().min() < 2:
        raise ValueError(
            "Combined hospital data must contain at least "
            "two examples of each Outcome class."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Fit the scaler on training data only.
    scaler = StandardScaler()

    scaler.fit(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Create local datasets for each hospital.
    clients = []

    for hospital_name, X_local, y_local in prepared_clients:

        local_train_indexes = X_local.index.intersection(
            X_train.index
        )

        if len(local_train_indexes) == 0:
            continue

        local_X = X_local.loc[local_train_indexes]
        local_y = y_local.loc[local_train_indexes]

        local_X_scaled = scaler.transform(local_X)

        client_x = torch.tensor(
            local_X_scaled,
            dtype=torch.float32
        )

        client_y = torch.tensor(
            local_y.to_numpy().reshape(-1, 1),
            dtype=torch.float32
        )

        clients.append(
            (hospital_name, client_x, client_y)
        )

    if not clients:
        raise ValueError(
            "No hospital has training data. Check your CSV files."
        )

    global_model = DiabetesNet()
    results = []

    progress = st.progress(0)

    for round_no in range(rounds):

        client_states = []
        client_sizes = []
        losses = []

        # Each hospital trains a local copy of the global model.
        for hospital_name, client_x, client_y in clients:

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

                loss = loss_function(
                    predictions,
                    client_y
                )

                loss.backward()
                optimizer.step()

            client_states.append(
                copy_weights(local_model)
            )

            client_sizes.append(len(client_x))
            losses.append(loss.item())

        # FedAvg: aggregate local hospital model weights.
        global_model.load_state_dict(
            average_weights(
                client_states,
                client_sizes
            )
        )

        # Evaluate the global model after aggregation.
        global_model.eval()

        with torch.no_grad():
            probabilities = global_model(
                torch.tensor(
                    X_test_scaled,
                    dtype=torch.float32
                )
            ).numpy().ravel()

        predicted = (
            probabilities >= 0.5
        ).astype(int)

        results.append({
            "Round": round_no + 1,
            "Accuracy": accuracy_score(
                y_test,
                predicted
            ),
            "Precision": precision_score(
                y_test,
                predicted,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                predicted,
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                predicted,
                zero_division=0
            ),
            "Mean Local Loss": float(
                np.mean(losses)
            )
        })

        progress.progress(
            (round_no + 1) / rounds,
            text=f"Completed federated round {round_no + 1}/{rounds}"
        )

        # Update graphs and results during training.
        current_results = pd.DataFrame(results)

        if chart_placeholder is not None:
            chart_placeholder.line_chart(
                current_results.set_index("Round")[
                    [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score"
                    ]
                ]
            )

        if table_placeholder is not None:
            table_placeholder.dataframe(
                current_results,
                use_container_width=True
            )

    return pd.DataFrame(results)


# ---------------------------------
# 7. SIDEBAR NAVIGATION
# ---------------------------------
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
st.caption(
    "Privacy-Preserving Federated Learning Demo"
)

st.warning(
    "Educational demonstration only. Hospital A, B and C are "
    "simulated clients inside this application. This is not a real "
    "multi-hospital deployment. The model is not clinically validated "
    "and must not be used for medical decisions."
)


# ---------------------------------
# 8. OVERVIEW
# ---------------------------------
if page == "Overview":

    st.header("Saved Model Performance")

    metrics = load_saved_metrics()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        get_metric(
            metrics,
            ["accuracy", "test_accuracy", "acc"]
        )
    )

    col2.metric(
        "Precision",
        get_metric(
            metrics,
            ["precision", "test_precision"]
        )
    )

    col3.metric(
        "Recall",
        get_metric(
            metrics,
            ["recall", "test_recall"]
        )
    )

    col4.metric(
        "F1 Score",
        get_metric(
            metrics,
            ["f1", "f1_score", "f1-score"]
        )
    )

    st.subheader("Project Workflow")

    st.write("1. Select participating hospitals.")
    st.write("2. Upload separate local CSV files if required.")
    st.write("3. Train each hospital's local model.")
    st.write("4. Aggregate model weights using FedAvg.")
    st.write("5. Monitor metrics after each federated round.")

    st.link_button(
        "Open Hugging Face Model",
        f"https://huggingface.co/{MODEL_REPO}"
    )

    if st.button("Refresh saved metrics"):
        load_saved_metrics.clear()
        st.rerun()


# ---------------------------------
# 9. FEDERATED TRAINING PAGE
# ---------------------------------
elif page == "Federated Training":

    st.header("🏥 Federated Hospital Training")

    st.info(
        "Choose Hospital A, B or C. Each selected hospital can use its "
        "own CSV file. If no CSV is uploaded, a portion of the public "
        "demo dataset is assigned to that simulated hospital."
    )

    selected_hospitals = st.multiselect(
        "Select hospitals participating in federated training",
        [
            "Hospital A",
            "Hospital B",
            "Hospital C"
        ],
        default=[
            "Hospital A",
            "Hospital B",
            "Hospital C"
        ]
    )

    hospital_data = {}

    if selected_hospitals:

        st.subheader("Local Hospital Datasets")

        st.caption(
            "Upload a separate CSV for each selected hospital. "
            "All CSV files must contain the same eight input features "
            "and the Outcome column."
        )

        for hospital in selected_hospitals:

            with st.expander(
                f"🏥 {hospital} — Local Data",
                expanded=True
            ):

                uploaded_file = st.file_uploader(
                    f"Upload CSV for {hospital}",
                    type=["csv"],
                    key=f"csv_{hospital}",
                    help=(
                        "Required columns: Pregnancies, Glucose, "
                        "BloodPressure, SkinThickness, Insulin, BMI, "
                        "DiabetesPedigreeFunction, Age, Outcome. "
                        "Outcome must be 0 or 1."
                    )
                )

                if uploaded_file is not None:

                    try:
                        local_df = pd.read_csv(
                            uploaded_file
                        )

                        hospital_data[hospital] = local_df

                        st.success(
                            f"{hospital}: {len(local_df)} rows loaded"
                        )

                        st.dataframe(
                            local_df.head(5),
                            use_container_width=True
                        )

                    except Exception as error:

                        st.error(
                            f"Could not read CSV: {error}"
                        )

                        hospital_data[hospital] = None

                else:

                    hospital_data[hospital] = None

                    st.caption(
                        "No local CSV uploaded. A partition of the "
                        "public demo dataset will be used."
                    )

    st.divider()

    st.subheader("Training Settings")

    col1, col2, col3 = st.columns(3)

    rounds = col1.slider(
        "Federated Rounds",
        1,
        10,
        5
    )

    epochs = col2.slider(
        "Local Epochs",
        1,
        10,
        3
    )

    learning_rate = col3.slider(
        "Learning Rate",
        0.001,
        0.1,
        0.01,
        step=0.001
    )

    if st.button(
        "▶ Start Federated Training",
        type="primary"
    ):

        if not selected_hospitals:

            st.error(
                "Select at least one hospital before training."
            )

        else:

            try:
                st.subheader("Live Training Metrics")

                chart_placeholder = st.empty()
                table_placeholder = st.empty()

                with st.spinner(
                    "Hospitals are training locally. "
                    "Metrics will update after each round..."
                ):

                    results = train_federated(
                        rounds=rounds,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        hospital_data=hospital_data,
                        chart_placeholder=chart_placeholder,
                        table_placeholder=table_placeholder
                    )

                st.success(
                    "Federated training completed successfully!"
                )

                latest = results.iloc[-1]

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Accuracy",
                    f"{latest['Accuracy']:.1%}"
                )

                c2.metric(
                    "Precision",
                    f"{latest['Precision']:.1%}"
                )

                c3.metric(
                    "Recall",
                    f"{latest['Recall']:.1%}"
                )

                c4.metric(
                    "F1 Score",
                    f"{latest['F1 Score']:.1%}"
                )

                st.subheader("Final Training Graph")

                st.line_chart(
                    results.set_index("Round")[
                        [
                            "Accuracy",
                            "Precision",
                            "Recall",
                            "F1 Score"
                        ]
                    ]
                )

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
                    results.to_csv(
                        index=False
                    ).encode("utf-8"),
                    "fedhealthops_results.csv",
                    "text/csv"
                )

                st.warning(
                    "This is a simulated federated-learning experiment. "
                    "The local data and model updates are processed within "
                    "this application; this does not establish a real "
                    "privacy-preserving connection between hospitals. "
                    "New training does not automatically overwrite the "
                    "saved Hugging Face model or create an MLflow run."
                )

            except Exception as error:

                st.error(
                    f"Training failed: {type(error).__name__}: {error}"
                )


# ---------------------------------
# 10. DEMO PREDICTION
# ---------------------------------
elif page == "Demo Prediction":

    st.header("Diabetes Model Demo")

    st.info(
        "Enter synthetic example values only. This model is for "
        "educational demonstrations and is not a medical diagnosis."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        pregnancies = col1.number_input(
            "Pregnancies",
            min_value=0,
            max_value=30,
            value=2
        )

        glucose = col1.number_input(
            "Glucose",
            min_value=0.0,
            value=120.0
        )

        bp = col1.number_input(
            "Blood Pressure",
            min_value=0.0,
            value=70.0
        )

        skin = col1.number_input(
            "Skin Thickness",
            min_value=0.0,
            value=20.0
        )

        insulin = col2.number_input(
            "Insulin",
            min_value=0.0,
            value=80.0
        )

        bmi = col2.number_input(
            "BMI",
            min_value=0.0,
            value=28.5
        )

        pedigree = col2.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=0.35
        )

        age = col2.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=33
        )

        threshold = st.slider(
            "Classification Threshold",
            0.1,
            0.9,
            0.5,
            0.05
        )

        submitted = st.form_submit_button(
            "Run Prediction",
            type="primary"
        )

    if submitted:

        try:
            model, scaler = load_saved_model()

            values = np.array([[
                pregnancies,
                glucose,
                bp,
                skin,
                insulin,
                bmi,
                pedigree,
                age
            ]], dtype=float)

            scaled_values = scaler.transform(values)

            with torch.no_grad():

                score = float(
                    model(
                        torch.tensor(
                            scaled_values,
                            dtype=torch.float32
                        )
                    ).item()
                )

            st.metric(
                "Model Output Score",
                f"{score:.1%}"
            )

            if score >= threshold:
                st.warning(
                    "The model predicts the positive class."
                )
            else:
                st.info(
                    "The model predicts the negative class."
                )

            st.caption(
                "Educational model output only. "
                "It must not be used for medical decisions."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {type(error).__name__}: {error}"
            )


# ---------------------------------
# 11. MODEL & MLOPS
# ---------------------------------
elif page == "Model & MLOps":

    st.header("Model Information")

    st.write("**Framework:** PyTorch")
    st.write("**Algorithm:** Federated Averaging (FedAvg)")
    st.write("**Neural Network:** 8 input features, 16 hidden units, 1 output")
    st.write("**Simulated Clients:** Hospital A, Hospital B and Hospital C")
    st.write("**Local Training:** Separate dataset per selected hospital")
    st.write("**Aggregation:** Weighted averaging of model parameters")
    st.write("**Experiment Tracking:** Existing MLflow notebook run")
    st.write("**Model Storage:** Hugging Face")

    st.link_button(
        "View Hugging Face Model",
        f"https://huggingface.co/{MODEL_REPO}"
    )

    st.warning(
        "Training from this page is an in-memory simulation. "
        "It does not automatically save the new global model to "
        "Hugging Face or log a new experiment to MLflow."
    )

    st.caption(
        "For a real multi-hospital deployment, each hospital would need "
        "its own secured training environment and a properly designed "
        "communication and privacy mechanism."
    )
