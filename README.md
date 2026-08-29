# Real-Time Network Intrusion Detection System (Random Forest)

A deployable mini-project built from the "Network Intrusion Detection System
Using Random Forest Algorithm" synopsis. It implements every phase from the
synopsis's methodology (dataset → preprocessing → train/test split → Random
Forest training → prediction → evaluation) and wraps it in a live,
interactive web dashboard you can run locally or deploy to the cloud in
minutes.

## What's inside

```
nids_project/
├── data/
│   └── generate_data.py   # Synthetic NSL-KDD-style dataset generator
├── model/                  # Created by train_model.py (trained artifacts)
├── train_model.py          # Preprocessing + Random Forest training + evaluation
├── app.py                  # Streamlit real-time detection dashboard
├── requirements.txt
├── Dockerfile               # For Render / Railway / any Docker host
└── README.md
```

## Why a synthetic dataset?

Real benchmark datasets (NSL-KDD, CICIDS2017, UNSW-NB15) require downloading
multi-hundred-MB files from external research repositories, which isn't
possible in a self-contained deployable package. `data/generate_data.py`
generates traffic with the **same feature schema and realistic statistical
behavior** as NSL-KDD (duration, protocol_type, service, flag, src_bytes,
dst_bytes, count, srv_count, error/service rates, dst_host stats) across 5
classes: `normal`, `dos`, `probe`, `r2l`, `u2r`.

**To swap in a real dataset later:** download NSL-KDD or CICIDS2017, save it
as `data/raw_dataset.csv`, and replace the call to
`generate_synthetic_dataset()` in `train_model.py` with `pd.read_csv(...)`
(matching column names to `CATEGORICAL_COLS` / `LABEL_COL`). No other code
needs to change.

## Run it locally

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates the model/ folder — only needed once)
python train_model.py

# 4. Launch the dashboard
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### What you'll see
- **Live Simulation** — a real-time feed of simulated packets classified
  as they "arrive," with live alerts, a running accuracy readout, and a
  bar chart of detected attack types.
- **Batch Upload (CSV)** — upload a CSV of traffic records and get bulk
  predictions plus a downloadable report.
- **Model Report** — accuracy/precision/recall/F1, confusion matrix,
  per-class report, feature importance, and ROC curves (matches the
  synopsis's Phase 6 evaluation requirements).

## Deploy it (free options)

### Option A — Streamlit Community Cloud (easiest, free, ~2 minutes)
1. Push this folder to a **public GitHub repo** (include the `model/`
   folder, or let the cloud build run `train_model.py` — see step 4).
2. Go to https://share.streamlit.io → **New app**.
3. Select your repo/branch and set the main file to `app.py`.
4. If you didn't commit the `model/` folder, add this as the app's
   "Advanced settings → Pre-run command" (or simply run
   `python train_model.py` locally once and commit the generated
   `model/*.pkl` and `model/*.json` files — simplest option).
5. Click **Deploy**. You'll get a public `https://<app-name>.streamlit.app` URL.

### Option B — Render.com (Docker, free tier available)
1. Push the folder (including `Dockerfile`) to a GitHub repo.
2. On Render: **New → Web Service** → connect the repo.
3. Render auto-detects the `Dockerfile`. Set:
   - **Port:** `8501`
4. Deploy — Render builds the image (which trains the model during build)
   and gives you a public URL.

### Option C — Hugging Face Spaces (free, Streamlit template)
1. Create a new Space → SDK: **Streamlit**.
2. Upload all files in this folder (or push via `git`).
3. Space auto-installs `requirements.txt` and runs `app.py`. Run
   `python train_model.py` once locally and commit the `model/` folder
   first, since Spaces doesn't run a custom build step by default.

### Option D — Any VPS / cloud VM with Docker
```bash
docker build -t nids-app .
docker run -p 8501:8501 nids-app
```
Then open `http://<server-ip>:8501`.

## Retraining / experimenting
- Adjust class balance or feature distributions in `data/generate_data.py`.
- Tune the Random Forest (`n_estimators`, `max_depth`, etc.) in
  `train_model.py`.
- Re-run `python train_model.py` — it overwrites `model/` with new
  artifacts and metrics, which the app picks up automatically.

## Mapping back to the synopsis
| Synopsis section | Where it lives |
|---|---|
| Phase 1: Dataset Collection | `data/generate_data.py` |
| Phase 2: Data Preprocessing | `train_model.py: preprocess()` |
| Phase 3: Train/Test Split (80/20) | `train_model.py` |
| Phase 4: Random Forest Construction | `train_model.py` (`RandomForestClassifier`) |
| Phase 5: Prediction | `train_model.py` + `app.py: predict_packets()` |
| Phase 6: Performance Evaluation | `model/metrics.json` + "Model Report" tab in `app.py` |
| Real-time detection (extension) | "Live Simulation" tab in `app.py` |
