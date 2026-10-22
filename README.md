

# FedHealthOps — Federated Health Intelligence

A student research demonstration featuring:
- Three simulated client datasets and FedAvg training rounds
- Evaluation metrics and a chart generated from the current training run
- Demo prediction using the saved PyTorch model and scaler from `affu678/fedhealthops-diabetes`

## Important limitations
- The three clients are simulated inside one app runtime; they are not separate hospitals.
- The training tab runs a new demonstration on a public dataset. It does not overwrite the hosted model repository or automatically log the new run to MLflow.
- Predictions are educational demonstrations only, not clinical advice or diagnosis. Do not enter real patient-identifiable information.

## Model artifacts
The app downloads `global_model.pth` and `scaler.joblib` from:
https://huggingface.co/affu678/fedhealthops-diabetes
