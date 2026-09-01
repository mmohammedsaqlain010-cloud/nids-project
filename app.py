"""
app.py
------
Real-Time Network Intrusion Detection System dashboard (Streamlit).

Two modes:
  1. Live Simulation - continuously generates realistic traffic "packets"
     (using the same statistical generators as training) and classifies
     them in real time with the trained Random Forest model, like a live
     SOC monitoring feed.
  2. Batch Upload - upload a CSV of network traffic records and get bulk
     predictions + a downloadable report.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud:
    1. Push this whole folder to a public/private GitHub repo
    2. Go to https://share.streamlit.io -> "New app"
    3. Point it at app.py in that repo -> Deploy
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from data.generate_data import (
    ATTACK_TYPES,
    ATTACK_WEIGHTS,
    _sample_categorical,
    SERVICES,
)

MODEL_DIR = Path("model")

st.set_page_config(
    page_title="Real-Time NIDS | Random Forest",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load model artifacts (cached so they only load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "rf_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, label_encoder, encoders, feature_columns, metrics


try:
    model, scaler, label_encoder, encoders, feature_columns, metrics = load_artifacts()
except FileNotFoundError:
    with st.spinner(
        "First-time setup: training the Random Forest model "
        "(takes about 10-20 seconds)..."
    ):
        from train_model import train_and_save
        train_and_save()
    load_artifacts.clear()  # clear the cached loader so it re-reads the new files
    model, scaler, label_encoder, encoders, feature_columns, metrics = load_artifacts()

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def make_one_packet(rng):
    """Generate a single random traffic record using the same underlying
    statistical model as training, then wrap it as a one-row DataFrame."""
    label = rng.choice(ATTACK_TYPES, p=ATTACK_WEIGHTS)

    if label == "normal":
        row = dict(
            duration=rng.exponential(50), src_bytes=rng.lognormal(6, 1.5),
            dst_bytes=rng.lognormal(6, 1.5), wrong_fragment=0, urgent=0,
            count=rng.poisson(5), srv_count=rng.poisson(5),
            serror_rate=rng.beta(1, 20), same_srv_rate=rng.beta(8, 2),
            diff_srv_rate=rng.beta(1, 10), dst_host_count=rng.poisson(20),
            dst_host_srv_count=rng.poisson(20),
            protocol_type=rng.choice(["tcp", "udp"], p=[0.7, 0.3]),
            service=rng.choice(SERVICES, p=[0.3, 0.1, 0.1, 0.2, 0.1, 0.05, 0.1, 0.05]),
            flag="SF",
        )
    elif label == "dos":
        row = dict(
            duration=rng.exponential(1), src_bytes=rng.lognormal(3, 1),
            dst_bytes=rng.lognormal(1, 1),
            wrong_fragment=rng.choice([0, 1], p=[0.7, 0.3]), urgent=0,
            count=rng.poisson(300), srv_count=rng.poisson(300),
            serror_rate=rng.beta(15, 2), same_srv_rate=rng.beta(9, 1),
            diff_srv_rate=rng.beta(1, 15), dst_host_count=rng.poisson(250),
            dst_host_srv_count=rng.poisson(250),
            protocol_type=rng.choice(["tcp", "icmp"], p=[0.6, 0.4]),
            service=rng.choice(["http", "private", "other"], p=[0.4, 0.4, 0.2]),
            flag=rng.choice(["S0", "REJ"], p=[0.7, 0.3]),
        )
    elif label == "probe":
        row = dict(
            duration=rng.exponential(5), src_bytes=rng.lognormal(2, 1),
            dst_bytes=rng.lognormal(1, 1), wrong_fragment=0, urgent=0,
            count=rng.poisson(40), srv_count=rng.poisson(5),
            serror_rate=rng.beta(3, 10), same_srv_rate=rng.beta(2, 8),
            diff_srv_rate=rng.beta(8, 2), dst_host_count=rng.poisson(150),
            dst_host_srv_count=rng.poisson(10),
            protocol_type=rng.choice(["tcp", "icmp"], p=[0.5, 0.5]),
            service=rng.choice(SERVICES), flag=rng.choice(["S0", "REJ", "SF"]),
        )
    elif label == "r2l":
        row = dict(
            duration=rng.exponential(20), src_bytes=rng.lognormal(4, 1),
            dst_bytes=rng.lognormal(4, 1), wrong_fragment=0,
            urgent=rng.choice([0, 1], p=[0.85, 0.15]),
            count=rng.poisson(3), srv_count=rng.poisson(3),
            serror_rate=rng.beta(2, 10), same_srv_rate=rng.beta(5, 5),
            diff_srv_rate=rng.beta(3, 7), dst_host_count=rng.poisson(10),
            dst_host_srv_count=rng.poisson(10), protocol_type="tcp",
            service=rng.choice(["ftp", "telnet", "smtp"], p=[0.4, 0.4, 0.2]),
            flag=rng.choice(["SF", "RSTR"], p=[0.6, 0.4]),
        )
    else:  # u2r
        row = dict(
            duration=rng.exponential(100), src_bytes=rng.lognormal(7, 2),
            dst_bytes=rng.lognormal(2, 2), wrong_fragment=0,
            urgent=rng.choice([0, 1], p=[0.5, 0.5]),
            count=rng.poisson(2), srv_count=rng.poisson(2),
            serror_rate=rng.beta(1, 15), same_srv_rate=rng.beta(6, 4),
            diff_srv_rate=rng.beta(2, 8), dst_host_count=rng.poisson(5),
            dst_host_srv_count=rng.poisson(5), protocol_type="tcp",
            service=rng.choice(["telnet", "ssh", "other"]), flag="SF",
        )

    row["_true_label"] = label  # kept only for the live demo accuracy readout
    return row


def predict_packets(df_raw):
    """Encode + scale + predict a batch of raw traffic records."""
    df = df_raw.copy()
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        # Map unseen categories to the first known class instead of crashing
        df[col] = df[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    X = df[feature_columns]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)
    pred_labels = label_encoder.inverse_transform(preds)
    confidence = probs.max(axis=1)
    return pred_labels, confidence


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ NIDS Control Panel")
mode = st.sidebar.radio("Mode", ["Live Simulation", "Batch Upload (CSV)", "Model Report"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** Random Forest (200 trees)\n\n"
    f"**Test Accuracy:** {metrics['accuracy']*100:.2f}%\n\n"
    f"**Classes:** {', '.join(metrics['class_names'])}"
)

# ---------------------------------------------------------------------------
# MODE 1: Live Simulation
# ---------------------------------------------------------------------------
if mode == "Live Simulation":
    st.title("🛡️ Real-Time Network Intrusion Detection")
    st.caption(
        "Simulates a live traffic feed and classifies every packet in real "
        "time using the trained Random Forest model."
    )

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        speed = st.slider("Packets / second", 1, 10, 3)
    with col_b:
        n_batches = st.slider("Total packets", 20, 500, 100, step=20)
    with col_c:
        run = st.button("▶ Start Live Detection", use_container_width=True)

    placeholder_metrics = st.empty()
    placeholder_alert = st.empty()
    placeholder_chart = st.empty()
    placeholder_table = st.empty()

    if run:
        rng = np.random.default_rng()
        log_rows = []
        counts = {c: 0 for c in metrics["class_names"]}
        correct = 0

        for i in range(n_batches):
            raw = make_one_packet(rng)
            true_label = raw.pop("_true_label")
            df_row = pd.DataFrame([raw])
            pred_label, conf = predict_packets(df_row)
            pred_label, conf = pred_label[0], conf[0]

            counts[pred_label] += 1
            if pred_label == true_label:
                correct += 1

            log_rows.append({
                "time_step": i + 1,
                "protocol": raw["protocol_type"],
                "service": raw["service"],
                "src_bytes": round(raw["src_bytes"], 1),
                "dst_bytes": round(raw["dst_bytes"], 1),
                "prediction": pred_label,
                "confidence": f"{conf*100:.1f}%",
                "status": "🟢 NORMAL" if pred_label == "normal" else "🔴 ATTACK",
            })

            with placeholder_metrics.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Packets analyzed", i + 1)
                m2.metric("Attacks detected", sum(v for k, v in counts.items() if k != "normal"))
                m3.metric("Normal traffic", counts["normal"])
                m4.metric("Live accuracy vs. ground truth", f"{correct/(i+1)*100:.1f}%")

            if pred_label != "normal" and conf > 0.6:
                placeholder_alert.error(
                    f"🚨 ALERT: {pred_label.upper()} attack detected "
                    f"(confidence {conf*100:.1f}%) — {raw['protocol_type']}/{raw['service']} traffic"
                )
            elif i % 10 == 0:
                placeholder_alert.success("✅ Network status: monitoring normally")

            chart_df = pd.DataFrame({"class": list(counts.keys()), "count": list(counts.values())})
            with placeholder_chart.container():
                st.bar_chart(chart_df.set_index("class"))

            table_df = pd.DataFrame(log_rows[-15:][::-1])
            placeholder_table.dataframe(table_df, use_container_width=True, hide_index=True)

            time.sleep(1 / speed)

        st.success(f"Simulation complete — {n_batches} packets processed.")
        full_log = pd.DataFrame(log_rows)
        st.download_button(
            "⬇ Download full detection log (CSV)",
            full_log.to_csv(index=False).encode(),
            file_name="nids_live_log.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# MODE 2: Batch Upload
# ---------------------------------------------------------------------------
elif mode == "Batch Upload (CSV)":
    st.title("📂 Batch Traffic Classification")
    st.caption(
        "Upload a CSV with the columns below (matching the training feature "
        "schema) to classify many records at once."
    )
    st.code(", ".join(feature_columns))

    sample = st.checkbox("Generate & preview a sample CSV I can upload")
    if sample:
        rng = np.random.default_rng(123)
        sample_rows = [make_one_packet(rng) for _ in range(10)]
        sample_df = pd.DataFrame(sample_rows).drop(columns=["_true_label"])
        st.dataframe(sample_df, use_container_width=True)
        st.download_button(
            "⬇ Download this as sample_traffic.csv",
            sample_df.to_csv(index=False).encode(),
            file_name="sample_traffic.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload traffic CSV", type=["csv"])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        missing = set(feature_columns) - set(df_raw.columns)
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            pred_labels, confidence = predict_packets(df_raw)
            result = df_raw.copy()
            result["prediction"] = pred_labels
            result["confidence"] = (confidence * 100).round(1)
            result["status"] = np.where(result["prediction"] == "normal", "NORMAL", "ATTACK")

            n_attacks = (result["status"] == "ATTACK").sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total records", len(result))
            c2.metric("Attacks flagged", int(n_attacks))
            c3.metric("Attack rate", f"{n_attacks/len(result)*100:.1f}%")

            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Download classified results",
                result.to_csv(index=False).encode(),
                file_name="nids_batch_results.csv",
                mime="text/csv",
            )

            st.subheader("Attack type breakdown")
            st.bar_chart(result["prediction"].value_counts())

# ---------------------------------------------------------------------------
# MODE 3: Model Report
# ---------------------------------------------------------------------------
else:
    st.title("📊 Model Performance Report")
    st.caption("Metrics computed on the held-out 20% test set during training.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{metrics['precision_weighted']*100:.2f}%")
    c3.metric("Recall", f"{metrics['recall_weighted']*100:.2f}%")
    c4.metric("F1-Score", f"{metrics['f1_weighted']*100:.2f}%")

    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"Actual: {c}" for c in metrics["class_names"]],
        columns=[f"Pred: {c}" for c in metrics["class_names"]],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Per-Class Report")
    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.subheader("Feature Importance (Random Forest)")
    fi_df = pd.DataFrame(
        list(metrics["feature_importance"].items()), columns=["feature", "importance"]
    ).set_index("feature")
    st.bar_chart(fi_df)

    st.subheader("ROC Curves (one-vs-rest)")
    roc_chart_df = pd.DataFrame({
        f"{cname} (AUC={data['auc']:.3f})": pd.Series(data["tpr"], index=data["fpr"])
        for cname, data in metrics["roc_data"].items()
    })
    st.line_chart(roc_chart_df)

    st.info(
        f"Trained on {metrics['n_train']} samples, tested on {metrics['n_test']} "
        "samples (80/20 split, stratified by class)."
    )
