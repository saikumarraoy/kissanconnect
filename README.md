# KissanConnect — Crop Disease Classifier

ML-powered crop disease detection feature for KissanConnect, built to gain real, verifiable frontend + ML experience.

## Project Phases

- **Phase 1 — Model Training** (`model_training/`): MobileNetV2 transfer learning on the PlantVillage dataset via Google Colab. Produces `kissanconnect_model.h5` + `class_names.json`.
- **Phase 2 — Backend API** (`backend/`): Flask `/predict` endpoint serving the trained model.
- **Phase 3 — Frontend** (`frontend/`): React UI for image upload + disease prediction results.
- **Disease Risk Advisory** (`docs/disease-risk-advisory.md`): planned rule-based weather-threshold risk engine (Wallin/BLITECAST-style for late blight, Zadoks model for stripe rust), paired with a diagnosis-logging feature to build real "same time last year" data over time. Scope/ordering (before or after the Flask API) still to be decided.

## Status

Phase 1 in progress — training notebook ready, not yet run.

## Structure

```
kissanconnect/
├── model_training/     # Colab notebook, dataset notes, saved model artifacts
├── backend/
│   ├── app/             # Flask application code
│   └── models/          # Trained model files (.h5) + class_names.json
├── frontend/
│   ├── src/              # React source
│   └── public/
└── docs/                # Planning notes, architecture decisions
```
