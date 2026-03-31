# ICU Capacity Forecasting Platform

This project predicts short-term ICU capacity risk and presents it in an operational dashboard.
It combines backend APIs, forecasting pipelines, risk logic, and a real-time UI to support proactive hospital decisions.

Deployed setup: **EC2 for application hosting** and **S3 for data storage workflows**.

## 📌 Project Overview

This system:

- Ingests ICU snapshots (`hospital_id`, `timestamp`, occupancy/capacity, operations fields)
- Stores snapshots, forecasts, and alerts in PostgreSQL
- Generates 24-step occupancy forecasts for each hospital
- Converts forecasted occupancy into risk levels (`GREEN`, `YELLOW`, `RED`)
- Displays:
  - current ICU KPIs
  - short-term forecast trend
  - active risk alerts
- Supports model experimentation in the ML module:
  - Baseline
  - SARIMA
  - Prophet

## 🧱 Architecture at a glance

- **Frontend (`frontend/`)**: React + Vite dashboard
- **Backend (`backend/`)**: FastAPI + SQLAlchemy + risk/forecast services
- **Database**: PostgreSQL (via Docker for local setup)
- **ML module (`ml/`)**: preprocessing, training, inference, evaluation
- **Cloud deployment**: EC2 runtime + S3 data storage path

## 📈 Key Results From Current Run

Hospital: `HOSPITAL_A`  
Evaluation horizon: `9` steps  
Train/Test split: `39 / 9`

| Model | MAE | RMSE | MAPE | SMAPE |
|---|---:|---:|---:|---:|
| Baseline | 1.611 | 1.893 | 1.867 | 1.886 |
| Prophet | 2.676 | 3.301 | 3.131 | 3.057 |
| SARIMA | 4.081 | 4.465 | 4.733 | 4.871 |

### Interpretation

- **Baseline performed best** on the current dataset.
- Advanced models did not outperform baseline in this sample, which can happen with limited or coarse-grained data.
- Current results validate end-to-end forecasting capability and model benchmarking flow.

## 🚦 Risk Logic (Operational Meaning)

Backend thresholds:
- `YELLOW` when occupancy ratio `>= 0.75`
- `RED` when occupancy ratio `>= 0.90`

Interpretation:
- `GREEN`: normal operating zone
- `YELLOW`: early pressure signal
- `RED`: critical pressure window requiring escalation

## 🧪 API Workflow (What runs in production path)

Core endpoints:
- `GET /health`
- `POST /snapshots`
- `GET /snapshots/latest/{hospital_id}`
- `POST /forecast/{hospital_id}`
- `GET /forecasts/{hospital_id}`
- `GET /alerts/{hospital_id}`

When `POST /forecast/{hospital_id}` is called:
1. hospital snapshot history is loaded
2. 24-step forecast is generated
3. risk level is computed per step
4. forecasts are stored
5. `YELLOW`/`RED` alerts are generated and stored

## 🖥 Dashboard Interpretation

Dashboard shows three layers of decision support:

1. **Current state** (KPI cards): occupancy, capacity, staffing, admissions/discharges
2. **Near-future trajectory** (forecast chart): predicted occupancy with threshold lines
3. **Action signals** (alerts panel): latest high-risk windows

Use the **Run Forecast + Refresh** button to trigger backend forecast generation and update all views.

## ☁️ EC2 + S3 Usage

- EC2 hosts the running application services.
- S3 is used for data storage workflows (raw data/object storage).
- Data can be seeded/loaded from S3 into DB in deployment workflows.
- Core dashboard/API contract remains DB-backed for fast online reads.

## 🏃 How to run locally

### 1) Start Postgres
```bash
docker compose up -d postgres
```

### 2) Install Python dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Run backend
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4) Run frontend
```bash
cd frontend
npm install
npm run dev
```

## 🔍 Sample API commands

### Health check
```bash
curl -s http://127.0.0.1:8000/health
```

Expected:
```json
{"status":"ok"}
```

### Run forecast
```bash
curl -s -X POST http://127.0.0.1:8000/forecast/HOSPITAL_A
```

### Get outputs
```bash
curl -s http://127.0.0.1:8000/forecasts/HOSPITAL_A
curl -s http://127.0.0.1:8000/alerts/HOSPITAL_A
```

## 🤖 ML module commands

### Train
```bash
python -m ml.train --input ml/data/raw/sample_hospital_a.csv --hospital-id HOSPITAL_A --model baseline
python -m ml.train --input ml/data/raw/sample_hospital_a.csv --hospital-id HOSPITAL_A --model sarima
python -m ml.train --input ml/data/raw/sample_hospital_a.csv --hospital-id HOSPITAL_A --model prophet
```

### Predict
```bash
python -m ml.predict --input ml/data/raw/sample_hospital_a.csv --hospital-id HOSPITAL_A --model baseline --horizon 24
```

### Evaluate
```bash
python -m ml.evaluate --input ml/data/raw/sample_hospital_a.csv --hospital-id HOSPITAL_A --write-results
```

## 🧩 Why this project matters

This project builds practical foundations for:
- healthcare capacity planning
- early-risk detection systems
- real-time alerting dashboards
- production ML lifecycle integration

It demonstrates full-stack delivery: ingestion -> forecast -> risk scoring -> API -> dashboard.

## 🛠 Current limitations and next upgrades

- Live backend currently uses a simple runtime forecast service.
- ML model integration into live API path is the next major upgrade.
- Dataset cadence improvements are in progress to improve forecast realism.
- Threshold and configuration unification across backend + ML is planned.