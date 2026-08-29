"""
generate_data.py
-----------------
Generates a synthetic network-traffic dataset that mimics the structure of
well-known intrusion detection benchmarks (NSL-KDD / CICIDS2017 / UNSW-NB15).

Why synthetic data?
Public benchmark datasets (NSL-KDD, CICIDS2017, UNSW-NB15) must be downloaded
from external repositories. To keep this project fully self-contained and
deployable without any external downloads, this script generates a dataset
with the SAME feature structure and statistical behaviour as those datasets.

If you want to use a real dataset instead:
1. Download NSL-KDD (https://www.unb.ca/cic/datasets/nsl.html) or
   CICIDS2017 (https://www.unb.ca/cic/datasets/ids-2017.html)
2. Place the CSV in data/raw_dataset.csv
3. Adjust FEATURE_COLUMNS / LABEL_COLUMN below to match the real column names
4. Skip calling generate_synthetic_dataset() and load your CSV directly in
   train_model.py

The generated features are modeled after real NIDS features:
duration, protocol_type, service, flag, src_bytes, dst_bytes, count,
srv_count, serror_rate, same_srv_rate, diff_srv_rate, dst_host_count,
dst_host_srv_count, wrong_fragment, urgent
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 20000

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "dns", "ssh", "telnet", "private", "other"]
FLAGS = ["SF", "S0", "REJ", "RSTR", "SH"]

ATTACK_TYPES = ["normal", "dos", "probe", "r2l", "u2r"]
# Roughly realistic class imbalance (attacks are rarer than normal traffic,
# and u2r/r2l are the rarest — matches real-world NIDS class distributions)
ATTACK_WEIGHTS = [0.55, 0.25, 0.12, 0.06, 0.02]


def _sample_categorical(rng, choices, n, weights=None):
    return rng.choice(choices, size=n, p=weights)


def generate_synthetic_dataset(n_samples=N_SAMPLES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    labels = _sample_categorical(rng, ATTACK_TYPES, n_samples, ATTACK_WEIGHTS)

    n = n_samples
    duration = np.zeros(n)
    src_bytes = np.zeros(n)
    dst_bytes = np.zeros(n)
    count = np.zeros(n)
    srv_count = np.zeros(n)
    serror_rate = np.zeros(n)
    same_srv_rate = np.zeros(n)
    diff_srv_rate = np.zeros(n)
    dst_host_count = np.zeros(n)
    dst_host_srv_count = np.zeros(n)
    wrong_fragment = np.zeros(n)
    urgent = np.zeros(n)
    protocol_type = np.empty(n, dtype=object)
    service = np.empty(n, dtype=object)
    flag = np.empty(n, dtype=object)

    for i, label in enumerate(labels):
        if label == "normal":
            duration[i] = rng.exponential(50)
            src_bytes[i] = rng.lognormal(6, 1.5)
            dst_bytes[i] = rng.lognormal(6, 1.5)
            count[i] = rng.poisson(5)
            srv_count[i] = rng.poisson(5)
            serror_rate[i] = rng.beta(1, 20)
            same_srv_rate[i] = rng.beta(8, 2)
            diff_srv_rate[i] = rng.beta(1, 10)
            dst_host_count[i] = rng.poisson(20)
            dst_host_srv_count[i] = rng.poisson(20)
            wrong_fragment[i] = 0
            urgent[i] = 0
            protocol_type[i] = rng.choice(["tcp", "udp"], p=[0.7, 0.3])
            service[i] = rng.choice(SERVICES, p=[0.3, 0.1, 0.1, 0.2, 0.1, 0.05, 0.1, 0.05])
            flag[i] = "SF"

        elif label == "dos":
            # DoS: huge counts, high error rate, short duration, flood-like
            duration[i] = rng.exponential(1)
            src_bytes[i] = rng.lognormal(3, 1)
            dst_bytes[i] = rng.lognormal(1, 1)
            count[i] = rng.poisson(300)
            srv_count[i] = rng.poisson(300)
            serror_rate[i] = rng.beta(15, 2)
            same_srv_rate[i] = rng.beta(9, 1)
            diff_srv_rate[i] = rng.beta(1, 15)
            dst_host_count[i] = rng.poisson(250)
            dst_host_srv_count[i] = rng.poisson(250)
            wrong_fragment[i] = rng.choice([0, 1], p=[0.7, 0.3])
            urgent[i] = 0
            protocol_type[i] = rng.choice(["tcp", "icmp"], p=[0.6, 0.4])
            service[i] = rng.choice(["http", "private", "other"], p=[0.4, 0.4, 0.2])
            flag[i] = rng.choice(["S0", "REJ"], p=[0.7, 0.3])

        elif label == "probe":
            # Probe/scan: many distinct services touched, low error, moderate count
            duration[i] = rng.exponential(5)
            src_bytes[i] = rng.lognormal(2, 1)
            dst_bytes[i] = rng.lognormal(1, 1)
            count[i] = rng.poisson(40)
            srv_count[i] = rng.poisson(5)
            serror_rate[i] = rng.beta(3, 10)
            same_srv_rate[i] = rng.beta(2, 8)
            diff_srv_rate[i] = rng.beta(8, 2)
            dst_host_count[i] = rng.poisson(150)
            dst_host_srv_count[i] = rng.poisson(10)
            wrong_fragment[i] = 0
            urgent[i] = 0
            protocol_type[i] = rng.choice(["tcp", "icmp"], p=[0.5, 0.5])
            service[i] = rng.choice(SERVICES)
            flag[i] = rng.choice(["S0", "REJ", "SF"])

        elif label == "r2l":
            # Remote-to-local: login/ftp-style attempts, small packets, some urgent flags
            duration[i] = rng.exponential(20)
            src_bytes[i] = rng.lognormal(4, 1)
            dst_bytes[i] = rng.lognormal(4, 1)
            count[i] = rng.poisson(3)
            srv_count[i] = rng.poisson(3)
            serror_rate[i] = rng.beta(2, 10)
            same_srv_rate[i] = rng.beta(5, 5)
            diff_srv_rate[i] = rng.beta(3, 7)
            dst_host_count[i] = rng.poisson(10)
            dst_host_srv_count[i] = rng.poisson(10)
            wrong_fragment[i] = 0
            urgent[i] = rng.choice([0, 1], p=[0.85, 0.15])
            protocol_type[i] = "tcp"
            service[i] = rng.choice(["ftp", "telnet", "smtp"], p=[0.4, 0.4, 0.2])
            flag[i] = rng.choice(["SF", "RSTR"], p=[0.6, 0.4])

        else:  # u2r
            # User-to-root: very rare, long sessions, unusual byte patterns
            duration[i] = rng.exponential(100)
            src_bytes[i] = rng.lognormal(7, 2)
            dst_bytes[i] = rng.lognormal(2, 2)
            count[i] = rng.poisson(2)
            srv_count[i] = rng.poisson(2)
            serror_rate[i] = rng.beta(1, 15)
            same_srv_rate[i] = rng.beta(6, 4)
            diff_srv_rate[i] = rng.beta(2, 8)
            dst_host_count[i] = rng.poisson(5)
            dst_host_srv_count[i] = rng.poisson(5)
            wrong_fragment[i] = 0
            urgent[i] = rng.choice([0, 1], p=[0.5, 0.5])
            protocol_type[i] = "tcp"
            service[i] = rng.choice(["telnet", "ssh", "other"])
            flag[i] = "SF"

    df = pd.DataFrame({
        "duration": duration,
        "protocol_type": protocol_type,
        "service": service,
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "wrong_fragment": wrong_fragment,
        "urgent": urgent,
        "count": count,
        "srv_count": srv_count,
        "serror_rate": serror_rate,
        "same_srv_rate": same_srv_rate,
        "diff_srv_rate": diff_srv_rate,
        "dst_host_count": dst_host_count,
        "dst_host_srv_count": dst_host_srv_count,
        "label": labels,
    })

    # shuffle rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    dataset = generate_synthetic_dataset()
    dataset.to_csv("data/network_traffic.csv", index=False)
    print(f"Generated {len(dataset)} rows -> data/network_traffic.csv")
    print(dataset["label"].value_counts())
