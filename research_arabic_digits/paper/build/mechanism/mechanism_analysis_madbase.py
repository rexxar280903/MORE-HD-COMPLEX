"""Analisis mekanistik EKSPLORATIF MADBase: mengapa dimensi readout tambahan MORE-HD-C kurang membantu?

Adaptasi langsung paper/build/mechanism_analysis.py (MNIST) untuk 180 run sel ablasi MADBase
(model A/B/C/D, fitur PCA/Hu/Zernike, K in {3, 6, 10}, seed 101..505). Tidak termasuk SAP beku.

Input : arsip mech_inputs.tar.gz (dibuat oleh pack_inputs.py di laptop) yang sudah diekstrak ke --src
Output: runs_mechanism.csv, jacobian_random.csv (di --out)
Usage : python mechanism_analysis_madbase.py --src <folder ekstrak> --out <folder> [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402

CODE = {"MORE-HD": "A", "MORE-HD-60P": "B", "MORE-HD-C-FixedRZ": "C", "MORE-HD-C": "D"}
OBS = ["IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
YODD = [i for i, o in enumerate(OBS) if o.count("Y") % 2 == 1]
YEVEN = [i for i in range(15) if i not in YODD]
EPS = 1e-10
N_JAC = 200
SRC: Path = Path(".")


def _core():
    sys.path.insert(0, str(SRC))
    from core.circuits import BatchedStatevector, build_initialization_bundle, build_model
    return BatchedStatevector, build_initialization_bundle, build_model


def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n < EPS, 0.0, v / np.maximum(n, EPS))


def nearest_label(vn, labels):
    ln = normalize(labels)
    d = 1.0 - vn @ ln.T
    d[np.linalg.norm(vn, axis=1) < EPS] = 1.0
    return np.argmin(d, axis=1)


def centroid_median(vn, y, K):
    return np.stack([normalize(np.median(vn[y == c], axis=0, keepdims=True))[0] for c in range(K)])


def min_sep_ratio(labels, K):
    ln = normalize(labels)
    d = 1.0 - ln @ ln.T
    iu = np.triu_indices(K, 1)
    return float(d[iu].min() / (1.0 + 1.0 / (K - 1)))


def fisher_ratio(vn, y, K):
    mu = vn.mean(axis=0)
    sb = sum((y == c).sum() * np.sum((vn[y == c].mean(axis=0) - mu) ** 2) for c in range(K))
    sw = sum(np.sum((vn[y == c] - vn[y == c].mean(axis=0)) ** 2) for c in range(K))
    return float(sb / sw)


def participation_ratio(x):
    ev = np.clip(np.linalg.eigvalsh(np.cov(x, rowvar=False)), 0, None)
    return float(ev.sum() ** 2 / np.sum(ev ** 2)) if ev.sum() > 0 else 0.0


def probe(vtr, ytr, vte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(vtr)
    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit(sc.transform(vtr), ytr)
    return float((clf.predict(sc.transform(vte)) == yte).mean())


def jacobian(engine, theta, phi):
    cols = []
    for j in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += np.pi / 2
        tm[j] -= np.pi / 2
        d = (engine.expectations(phi, engine.unitary_columns(tp)) -
             engine.expectations(phi, engine.unitary_columns(tm))) / 2.0
        cols.append(d.reshape(-1))
    return np.stack(cols, axis=1)


def spectrum_metrics(J, prefix):
    s = np.linalg.svd(J, compute_uv=False)
    smax = s[0] if s.size and s[0] > 0 else 1.0
    p = s / s.sum() if s.sum() > 0 else s
    p = p[p > 0]
    lam = s ** 2
    return {
        f"{prefix}_rank": int(np.sum(s > 1e-6 * smax)),
        f"{prefix}_eff_rank": float(np.exp(-np.sum(p * np.log(p)))),
        f"{prefix}_pr": float(lam.sum() ** 2 / np.sum(lam ** 2)),
        f"{prefix}_n90": int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.90) + 1),
        f"{prefix}_colnorm_min": float(np.linalg.norm(J, axis=0).min()),
    }


def analyse(key: str) -> dict:
    BatchedStatevector, build_initialization_bundle, build_model = _core()
    rd = SRC / "runs" / key
    cfg = json.loads((rd / "config.json").read_text())
    arch, K, seed, feat = cfg["architecture"], int(cfg["n_classes"]), int(cfg["seed"]), cfg["feature_method"]
    cell = SRC / "cells" / f"{feat}__K{K}__S{seed}"
    model = build_model(arch, build_initialization_bundle(seed))
    eng = BatchedStatevector(model)
    Xtr, ytr = np.load(cell / "X_train_scaled.npy"), np.load(cell / "y_train.npy")
    Xte, yte = np.load(cell / "X_test_scaled.npy"), np.load(cell / "y_test.npy")
    theta = np.load(rd / "artifacts" / "supervised_params_final.npy")
    ql = json.loads((rd / "artifacts" / "quantum_labels.json").read_text())
    labels = np.array([[ql["labels"][str(c)][o] for o in OBS] for c in range(K)])
    vtr, vte = eng.outputs(Xtr, theta), eng.outputs(Xte, theta)
    ntr, nte = normalize(vtr), normalize(vte)
    pred = nearest_label(nte, labels)
    metrics = json.loads((rd / "logs" / "metrics_final.json").read_text())
    rec = {
        "run": key, "model": CODE[arch], "feature": feat, "K": K, "seed": seed,
        "acc_logged": metrics["accuracy"], "acc_reproduced": float((pred == yte).mean()),
        "sep_ratio_labels": min_sep_ratio(labels, K),
        "fisher_ratio": fisher_ratio(nte, yte, K),
        "pr_readout": participation_ratio(nte),
        "probe_all15": probe(ntr, ytr, nte, yte),
        "probe_yeven9": probe(ntr[:, YEVEN], ytr, nte[:, YEVEN], yte),
        "yodd_energy_test": float(np.mean(np.sum(nte[:, YODD] ** 2, axis=1))),
    }
    oracle = centroid_median(ntr, ytr, K)
    rec["acc_oracle_labels"] = float((nearest_label(nte, oracle) == yte).mean())
    rec["sep_ratio_oracle"] = min_sep_ratio(oracle, K)
    if arch in ("MORE-HD-C", "MORE-HD-C-FixedRZ"):
        rec["acc_labels_yeven_only"] = float((nearest_label(normalize(vte[:, YEVEN]), labels[:, YEVEN]) == yte).mean())
    idx = np.concatenate([np.where(yte == c)[0][: N_JAC // K + 1] for c in range(K)])[:N_JAC]
    phi = eng.encode(Xte[idx])
    J = jacobian(eng, theta, phi)
    rec.update(spectrum_metrics(J, "jac"))
    if arch == "MORE-HD-C":
        rz_cols = [l * 20 + 10 + q for l in range(3) for q in range(10)]
        ry_cols = [l * 20 + q for l in range(3) for q in range(10)]
        rec.update(spectrum_metrics(J[:, ry_cols], "jacRYonly"))
        rec["rz_colnorm_share"] = float(np.linalg.norm(J[:, rz_cols]) ** 2 / np.linalg.norm(J) ** 2)
        last_data_rz = [2 * 20 + 10 + q for q in range(8)]
        rec["last_layer_data_rz_colnorm_max"] = float(np.linalg.norm(J[:, last_data_rz], axis=0).max())
    if arch == "MORE-HD-60P":
        rec.update(spectrum_metrics(J[:, :30], "jacFirst30"))
    return rec


def random_jacobians(n_draws=12):
    BatchedStatevector, build_initialization_bundle, build_model = _core()
    cell = SRC / "cells" / "PCA__K10__S101"
    Xte, yte = np.load(cell / "X_test_scaled.npy"), np.load(cell / "y_test.npy")
    idx = np.concatenate([np.where(yte == c)[0][:20] for c in range(10)])
    rows = []
    rng = np.random.default_rng(20261011)
    for arch, code in CODE.items():
        model = build_model(arch, build_initialization_bundle(101))
        eng = BatchedStatevector(model)
        phi = eng.encode(Xte[idx])
        for d in range(n_draws):
            theta = rng.uniform(0, 2 * np.pi, model.n_trainable)
            J = jacobian(eng, theta, phi)
            v = eng.expectations(phi, eng.unitary_columns(theta))
            r = {"model": code, "draw": d, "n_params": model.n_trainable, "pr_outputs_random": participation_ratio(normalize(v))}
            r.update(spectrum_metrics(J, "jac"))
            rows.append(r)
    return rows


def _init(src):
    global SRC
    SRC = Path(src)


def main():
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    SRC = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(p.name for p in (SRC / "runs").iterdir())
    if args.limit:
        keys = keys[: args.limit]
    print(f"{len(keys)} runs", flush=True)
    import pandas as pd
    recs = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(SRC),)) as ex:
        futs = [ex.submit(analyse, k) for k in keys]
        rnd = ex.submit(random_jacobians)
        for i, f in enumerate(futs, 1):
            recs.append(f.result())
            if i % 10 == 0:
                print(f"  {i}/{len(futs)}", flush=True)
                pd.DataFrame(recs).to_csv(out / "runs_mechanism.partial.csv", index=False)
        rnd_rows = rnd.result()
    df = pd.DataFrame(recs).sort_values(["feature", "K", "seed", "model"])
    df.to_csv(out / "runs_mechanism.csv", index=False)
    pd.DataFrame(rnd_rows).to_csv(out / "jacobian_random.csv", index=False)
    print("max |logged - reproduced accuracy| =", (df.acc_logged - df.acc_reproduced).abs().max(), flush=True)


if __name__ == "__main__":
    main()
