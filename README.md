# WattSense: Smart Energy Consumption Prediction

WattSense is a web application that helps households understand and anticipate their electricity use. It shows a household's hourly consumption, forecasts the next 24 hours, 7 days and 30 days with a bidirectional CNN-LSTM model, and detects unusual consumption with an Isolation Forest and LSTM autoencoder ensemble.

The system runs as three Docker containers:

| Service   | Technology            | Address               |
| Front end | React 18 + Vite       | http://localhost:5173 |
| API       | FastAPI (Python 3.11) | http://localhost:8000 (interactive docs at http://localhost:8000/docs) |
| Database  | PostgreSQL 16         | localhost:5432        |

---

## 1. Prerequisites

- **Docker Desktop** (Windows or macOS) or **Docker Engine with Docker Compose v2** (Linux). Make sure Docker is running before you start.
- **At least 8 GB of RAM and about 10 GB of free disk space.** The API image includes TensorFlow 
- **Ports 5173, 8000 and 5432 free** on your machine.

---

## 2. Large files

The application needs a few large files. Check that they are in the folders below. If they are already there in the copy you received, skip to Section 3. Without them the application still starts, but some features will not work.

| File(s)                                  | Place them in    | Needed for |
| `smart_energy.dump` (SGSC database dump) | `infra/init-db/` | The 30 SGSC households used by existing users |
| `bidirectional_cnn_lstm_pooled.keras` and `bidirectional_direct_multistep_1month.keras` | `backend/ml/forecasting/cnn_bidirectional/saved_models/` | Forecasting |
| Anomaly model files (`isolation_forest.pkl`, `if_scaler.pkl`, `if_preprocess.pkl`, `if_threshold.pkl`, `ae_model.keras`, `ae_scaler.pkl`, `ae_preprocess.pkl`, `ae_threshold.pkl`, `if_ae_ensemble.pkl`) | `backend/ml/outputs/models/` | The anomaly check |


## 3. First-time setup

### Step 1: Extract the project

Extract the zip file to a folder on your computer, for example `Documents/wattsense`.Then open a terminal (PowerShell on Windows) in the extracted folder, the one that contains `docker-compose.yml`:


### Step 2: Build and start everything

```bash
docker compose up --build
```

The first build takes several minutes. When the logs settle, check:

- API health: http://localhost:8000/health
- API and database health: http://localhost:8000/health/db
- The application: http://localhost:5173

---

## 4. Loading the demonstration data (recommended)

The SGSC data ends around 2014, so it cannot show forecasts based on current weather. Two synthetic demonstration households with a year of recent hourly data are included: one with normal usage and one with four planted anomalies.

With the containers running, open a second terminal in the project root and run:

```bash
docker compose exec backend python scripts/seed_demo_households.py --create-demo-accounts
```

This loads both households, fetches matching weather, and creates two ready-made logins:

| Email | Password | Shows |
| --- | --- | --- |
| demo.clean@example.com | demo1234 | Household with normal usage |
| demo.anomalies@example.com | demo1234 | Household with four planted anomalies |

You can choose a different password with `--demo-password`. The script is safe to run again.

These are local demonstration accounts only.

## 5. Using the application

Open http://localhost:5173.

### Existing user (a household already in the database)

1. Log in with a demonstration account (Section 4).
2. The dashboard opens on the last 30 days. Change the range with the preset buttons or the From and To dates.
3. The forecast builds automatically on the first visit. It takes 1 to 4 minutes, and progress is shown on screen.
4. Choose the forecast horizon (24 hours, 7 days or 30 days), start the live meter demonstration, run the anomaly check, or download a CSV or PDF report.

### New user (uploads their own data)

1. Click **Register** and create an account. A newly registered account is treated as a new user.
2. Upload a CSV file of hourly consumption with two columns, `timestamp` and `kwh`, one row per hour:

   ```csv
   timestamp,kwh
   2026-07-15 00:00:00,0.173
   2026-07-15 01:00:00,0.211
   ```

   - At least 240 hourly rows (10 days); about 30 days or more for the 30-day forecast.
   - At most 15 MB.
   - The data should **end within the last few days**. Older data still works, but the forecast then uses a fixed weather value and the dashboard shows a warning.
   - A ready-made sample file is provided at `data/samples/new_user_sample_90days.csv`.
3. Enter your location (for example `Sydney, Australia`), country code and state, then submit.
4. Wait for the forecast to finish (1 to 4 minutes). You can close the page and come back.


## 6. Team

| Area                                         | Owner   |
| Data engineering                             | Alka    |
| Forecasting                                  | Binita  |
| Feature extraction and anomaly detection     | Mishael |
| visualisation and user management            | Jasim   |
| Recommendation engine and billing estimation | Abrein  |
