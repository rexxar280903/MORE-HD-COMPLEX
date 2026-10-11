#!/usr/bin/env python3
"""Build compact, browser-friendly data files for the MORE-HD dashboard — studi MADBase (digit Arab).

Salinan terpisah dari dashboard MNIST (research_data/data_from_laptop/dashboard); file MNIST tidak disentuh.
Sumber: research_arabic_digits/research_data/{consolidated,analysis} dan research_arabic_digits/paper/build
(numbers.json, jalurB_summary.csv, confusion_pairs_k10.csv, mechanism/runs_mechanism.csv) untuk tab temuan.

The source workbooks are intentionally treated as read-only.  The parser uses
the Python standard library and streams worksheet XML so the 35–96 MB XLSX
files can be processed on a laptop without loading an entire workbook into
memory.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"
RUN_OUT = OUT / "runs"
ANALYSIS = ROOT / "analysis"
CONSOLIDATED = ROOT / "consolidated"
PAPER_BUILD = ROOT.parent / "paper" / "build"

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

WORKBOOKS = [
    ("PRIMARY", CONSOLIDATED / "primary_240runs.xlsx"),
    ("ABLATION", CONSOLIDATED / "ablation_90runs.xlsx"),
    ("MORE_REFERENCE", CONSOLIDATED / "more_reference_120runs.xlsx"),
]

RUN_FIELDS = [
    "status", "run_mode", "K", "classes", "feature_method", "architecture",
    "model_code", "seed", "replication_id", "n_params", "n_train_total",
    "n_val_total", "n_test_total", "cluster_nfev", "final_cluster_loss",
    "cluster_final_point_eval_id", "cluster_best_observed_eval_id",
    "cluster_n_optimization_evals", "cluster_optimizer_success",
    "cluster_optimizer_status", "pseudo_accuracy_val", "avg_margin_val",
    "min_separation", "min_separation_ratio", "min_separation_val_ratio",
    "correlation_consistency", "active_dimensions", "yodd_norm_fraction",
    "yodd_norm_fraction_supervised", "supervised_nfev",
    "final_supervised_train_loss", "final_supervised_val_loss",
    "supervised_final_point_eval_id", "supervised_best_observed_eval_id",
    "supervised_n_optimization_evals", "supervised_optimizer_success",
    "supervised_optimizer_status", "accuracy", "precision_macro",
    "recall_macro", "f1_macro", "n_correct", "n_wrong",
    "total_runtime_sec", "total_cpu_sec", "data_pipeline_sec", "setup_sec",
    "clustering_loop_sec", "quantum_label_extraction_sec",
    "supervised_loop_sec", "final_evaluation_sec", "structural_diagnostics_sec",
    "n_parallel_declared", "n_degenerate_test_outputs", "started_at",
    "finished_at", "n_trainable_params", "n_fixed_rz_params",
    "n_variational_layers", "variational_gate_count", "entangling_block_count",
    "cnot_count", "circuit_depth", "backend_max_abs_diff",
]

CLUSTER_FIELDS = [
    "eval_id", "phase", "train_loss", "pseudo_accuracy_val", "avg_margin_val",
    "mean_true_distance_val", "min_separation_val", "min_separation",
    "min_separation_ratio", "min_separation_val_ratio", "correlation_consistency",
    "active_dimensions", "yodd_norm_fraction", "eval_runtime_sec",
    "n_degenerate_cluster_outputs", "n_degenerate_val_outputs",
]

SUPERVISED_FIELDS = [
    "eval_id", "phase", "train_loss", "val_loss", "pseudo_accuracy_val",
    "avg_margin_val", "eval_runtime_sec", "n_degenerate_train_outputs",
    "n_degenerate_val_outputs",
]

DETAIL_SHEETS = [
    ("02_Run_Config", "config", [
        "run_mode", "K", "classes", "n_train_per_class", "n_val_per_class",
        "n_test_per_class", "architecture", "feature_method", "n_data_qubits",
        "n_readout_qubits", "n_input_channels", "pca_n_components", "hu_raw_dim",
        "hu_padding_value", "zernike_terms", "cobyla_tol", "cobyla_rhobeg",
        "max_nfev_clustering", "max_nfev_supervised", "scipy_version", "master_seed",
        "data_seed", "ry_core_seed", "rz_phase_seed", "pair_seed", "label_seed",
        "split_manifest", "paired_split_key", "nested_class_split",
        "active_dim_threshold", "n_params", "subseed_rule", "numpy_version",
        "pennylane_version", "simulation_mode", "shots", "sim_dtype", "device_name",
        "eps_norm", "centroid_rule", "centroid_source", "val_monitor_policy",
        "n_val_monitor_per_class", "correlation_rule", "pca_svd_solver",
    ]),
    ("05_Correlation_Matrix", "correlation", [
        "class_i", "class_j", "raw_mse", "S_value",
    ]),
    ("06_Quantum_Labels", "quantum_labels", [
        "class", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX",
        "YY", "YZ", "ZI", "ZX", "ZY", "ZZ", "X", "Y", "Z",
        "n_label_samples", "n_degenerate_samples_excluded", "label_prenorm",
        "label_degenerate",
    ]),
    ("07_Pairwise_Distance", "pairwise_distance", [
        "class_i", "class_j", "S_value", "cosine_distance", "simplex_bound",
        "separation_ratio",
    ]),
    ("08_Dimension_Activity", "dimension_activity", [
        "checkpoint", "observable", "is_yodd", "mean_abs", "max_abs", "std_abs",
        "active_threshold", "is_active", "norm_fraction",
    ]),
    ("11_Test_Predictions", "test_predictions", [
        "sample_id", "source_sample_id", "source_index", "true_class", "pred_class",
        "distance_true", "distance_pred", "margin", "correct",
    ]),
]

INT_FIELDS = {
    "K", "seed", "n_params", "n_train_total", "n_val_total", "n_test_total",
    "cluster_nfev", "cluster_final_point_eval_id", "cluster_best_observed_eval_id",
    "cluster_n_optimization_evals", "active_dimensions", "supervised_nfev",
    "supervised_final_point_eval_id", "supervised_best_observed_eval_id",
    "supervised_n_optimization_evals", "n_correct", "n_wrong", "n_parallel_declared",
    "n_degenerate_test_outputs", "n_trainable_params", "n_fixed_rz_params",
    "n_variational_layers", "variational_gate_count", "entangling_block_count",
    "cnot_count", "circuit_depth", "eval_id", "n_degenerate_cluster_outputs",
    "n_degenerate_val_outputs", "n_degenerate_train_outputs", "support", "correct",
    "wrong", "class", "actual_class", "predicted_class", "count", "n_active",
    "n_active_yodd", "gate_count", "depth", "n_warmup", "n_repeat", "batch_size",
    "n_seeds", "n_pairs", "a_right_b_wrong", "a_wrong_b_right",
    "n_train_per_class", "n_val_per_class", "n_test_per_class", "n_data_qubits",
    "n_readout_qubits", "n_input_channels", "pca_n_components", "hu_raw_dim",
    "zernike_terms", "max_nfev_clustering", "max_nfev_supervised", "master_seed",
    "data_seed", "ry_core_seed", "rz_phase_seed", "pair_seed", "label_seed",
    "n_val_monitor_per_class", "class_i", "class_j", "n_label_samples",
    "n_degenerate_samples_excluded", "label_degenerate", "is_yodd", "is_active",
    "sample_id", "source_index", "true_class", "pred_class", "correct",
}

FLOAT_FIELDS = {
    "final_cluster_loss", "pseudo_accuracy_val", "avg_margin_val", "min_separation",
    "min_separation_ratio", "min_separation_val_ratio", "correlation_consistency",
    "yodd_norm_fraction", "yodd_norm_fraction_supervised",
    "final_supervised_train_loss", "final_supervised_val_loss", "accuracy",
    "precision_macro", "recall_macro", "f1_macro", "total_runtime_sec",
    "total_cpu_sec", "data_pipeline_sec", "setup_sec", "clustering_loop_sec",
    "quantum_label_extraction_sec", "supervised_loop_sec", "final_evaluation_sec",
    "structural_diagnostics_sec", "backend_max_abs_diff", "train_loss", "val_loss",
    "mean_true_distance_val", "min_separation_val", "eval_runtime_sec", "precision",
    "recall", "f1", "threshold", "wall_median_sec", "wall_q1_sec", "wall_q3_sec",
    "wall_p95_sec", "cpu_median_sec", "unitary_median_sec", "accuracy_mean",
    "accuracy_sd", "accuracy_median", "accuracy_iqr", "f1_macro_mean",
    "f1_macro_sd", "f1_macro_median", "f1_macro_iqr", "min_separation_ratio_mean",
    "min_separation_ratio_sd", "min_separation_ratio_median",
    "min_separation_ratio_iqr", "min_separation_mean", "min_separation_sd",
    "min_separation_median", "min_separation_iqr", "min_separation_val_ratio_mean",
    "min_separation_val_ratio_sd", "min_separation_val_ratio_median",
    "min_separation_val_ratio_iqr", "yodd_norm_fraction_mean",
    "yodd_norm_fraction_sd", "yodd_norm_fraction_median",
    "yodd_norm_fraction_iqr", "runtime_mean_sec", "runtime_median_sec",
    "runtime_iqr_sec", "mean_delta", "sd_delta", "median_delta", "iqr_delta",
    "ci95_low", "ci95_high", "hedges_g", "sign_flip_p", "sign_flip_p_holm",
    "p_value", "p_value_holm_within_seed", "accuracy_merged_6_9",
    "six_nine_confusion_rate", "share_of_errors_6_9", "delta_accuracy", "delta_f1",
    "hu_padding_value", "cobyla_tol", "cobyla_rhobeg", "active_dim_threshold",
    "shots", "eps_norm", "raw_mse", "S_value", "IX", "IY", "IZ", "XI", "XX",
    "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ", "X", "Y",
    "Z", "label_prenorm", "cosine_distance", "simplex_bound", "separation_ratio",
    "mean_abs", "max_abs", "std_abs", "norm_fraction", "distance_true",
    "distance_pred", "margin",
}


def col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    result = 0
    for ch in letters:
        result = result * 26 + ord(ch.upper()) - 64
    return result - 1


def parse_value(field: str, value: object) -> object:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value or value.lower() in {"nan", "none", "n/a", "na"}:
        return None
    try:
        if field in INT_FIELDS:
            return int(float(value))
        if field in FLOAT_FIELDS:
            result = float(value)
            return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None
    return value


def sheet_targets(book: zipfile.ZipFile) -> Dict[str, str]:
    wb = ET.fromstring(book.read("xl/workbook.xml"))
    rels_xml = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    rels = {node.attrib["Id"]: node.attrib["Target"] for node in rels_xml}
    result: Dict[str, str] = {}
    sheets = wb.find(f"{{{NS_MAIN}}}sheets")
    if sheets is None:
        return result
    for node in sheets:
        rid = node.attrib[f"{{{NS_REL}}}id"]
        target = rels[rid].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        result[node.attrib["name"]] = target
    return result


def iter_sheet(path: Path, sheet_name: str) -> Iterator[Dict[str, object]]:
    """Yield rows from an inline-string XLSX worksheet, using row 3 as header."""
    with zipfile.ZipFile(path) as book:
        target = sheet_targets(book).get(sheet_name)
        if not target:
            return
        headers: Dict[int, str] = {}
        with book.open(target) as stream:
            for _event, row in ET.iterparse(stream, events=("end",)):
                if not row.tag.endswith("}row"):
                    continue
                row_number = int(row.attrib.get("r", "0") or 0)
                cells: Dict[int, str] = {}
                for cell in list(row):
                    if not cell.tag.endswith("}c"):
                        continue
                    idx = col_index(cell.attrib.get("r", "A1"))
                    if cell.attrib.get("t") == "inlineStr":
                        text = "".join(
                            child.text or "" for child in cell.iter()
                            if child.tag.endswith("}t")
                        )
                    else:
                        value_node = next(
                            (child for child in cell if child.tag.endswith("}v")), None
                        )
                        text = "" if value_node is None else (value_node.text or "")
                    cells[idx] = text
                if row_number == 3:
                    headers = {idx: value for idx, value in cells.items() if value}
                elif row_number > 3 and headers:
                    record = {
                        name: parse_value(name, cells.get(idx, ""))
                        for idx, name in headers.items()
                    }
                    if any(value is not None for value in record.values()):
                        yield record
                row.clear()


def normalize_run(track: str, row: Dict[str, object]) -> Dict[str, object]:
    condition_id = row.get("condition_id") or row.get("ablation_condition_id") or row.get("reference_condition_id")
    run_uid = row.get("run_uid") or row.get("ablation_run_uid") or row.get("reference_run_uid")
    architecture = str(row.get("architecture") or "Unknown")
    model_code = row.get("model_code")
    if not model_code:
        model_code = {"MORE-HD": "A", "MORE-HD-C": "D", "MORE-REPRO": "M"}.get(architecture, architecture)
    result = {
        "track": track,
        "condition_id": condition_id,
        "run_uid": run_uid,
        "history_key": f"{track}__{run_uid}",
        "model_code": model_code,
        "architecture": architecture,
    }
    for field in RUN_FIELDS:
        if field in {"model_code", "architecture"}:
            continue
        if field in row:
            result[field] = row[field]
    result["history_file"] = f"data/runs/{safe_name(result['history_key'])}.json"
    return result


def normalize_identity(track: str, row: Dict[str, object]) -> Dict[str, object]:
    condition_id = row.get("condition_id") or row.get("ablation_condition_id") or row.get("reference_condition_id")
    run_uid = row.get("run_uid") or row.get("ablation_run_uid") or row.get("reference_run_uid")
    architecture = str(row.get("architecture") or "Unknown")
    model_code = row.get("model_code") or {"MORE-HD": "A", "MORE-HD-C": "D", "MORE-REPRO": "M"}.get(architecture, architecture)
    return {
        "track": track,
        "condition_id": condition_id,
        "run_uid": run_uid,
        "history_key": f"{track}__{run_uid}",
        "model_code": model_code,
        "architecture": architecture,
    }


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def write_json(path: Path, payload: object, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            separators=None if pretty else (",", ":"),
            indent=2 if pretty else None,
            allow_nan=False,
        )


def read_csv(name: str) -> List[Dict[str, object]]:
    path = ANALYSIS / name
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: parse_value(key, value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_findings() -> Dict[str, object]:
    """Ringkasan temuan dari research_arabic_digits/paper/build (dibuat oleh analyze_madbase.py)."""
    out: Dict[str, object] = {}
    numbers = PAPER_BUILD / "numbers.json"
    if numbers.exists():
        num = json.loads(numbers.read_text(encoding="utf-8"))
        keep = ["primary_acc", "primary_acc_by_feature", "primary_f1_macro", "primary_min_separation_ratio", "ablation",
                "mcnemar", "model_mean_acc", "model_mean_acc_ablation_cells", "baseline_mean", "more_A_minus_M",
                "more_D_minus_M", "six_nine", "jalurB", "cross", "mechanism", "confusion_top3", "admission", "quality"]
        out["numbers"] = {key: num.get(key) for key in keep if key in num}
    for name, key in (("jalurB_summary.csv", "jalurB"), ("confusion_pairs_k10.csv", "confusion_pairs"),
                      ("cross_dataset_cells.csv", "cross_cells")):
        path = PAPER_BUILD / name
        if path.exists():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                out[key] = [{k: (float(v) if v not in ("", None) and _is_number(v) else v) for k, v in row.items()}
                            for row in csv.DictReader(handle)]
    return out


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def extract_histories(
    track: str,
    workbook: Path,
    sheet_name: str,
    fields: List[str],
    part_name: str,
) -> int:
    current_key: Optional[str] = None
    current_meta: Optional[Dict[str, object]] = None
    rows: List[List[object]] = []
    run_count = 0

    def flush() -> None:
        nonlocal run_count, rows, current_key, current_meta
        if current_key is None or current_meta is None:
            return
        filename = RUN_OUT / f"{safe_name(current_key)}.json"
        if filename.exists():
            with filename.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = {"meta": current_meta}
        payload[part_name] = {"columns": fields, "rows": rows}
        write_json(filename, payload)
        run_count += 1
        if run_count % 20 == 0:
            print(f"    {sheet_name}: {run_count} run", flush=True)
        rows = []

    for row in iter_sheet(workbook, sheet_name):
        identity = normalize_identity(track, row)
        key = identity["history_key"]
        if current_key is not None and key != current_key:
            flush()
        if key != current_key:
            current_key = key
            current_meta = {
                **identity,
                "K": row.get("K"),
                "feature_method": row.get("feature_method"),
                "seed": row.get("seed"),
            }
        rows.append([row.get(field) for field in fields])
    flush()
    return run_count


def main(phase: str = "all", only_track: Optional[str] = None, only_part: Optional[str] = None) -> int:
    started = time.time()
    print("Menyiapkan data dashboard MORE-HD (MADBase)...", flush=True)
    # Jangan hapus direktori keluaran saat server lokal aktif: pada Windows,
    # browser dapat menahan handle pada folder tersebut. Setiap berkas yang
    # dipakai dashboard ditulis ulang secara deterministik; berkas lama yang
    # tidak lagi dirujuk manifest tidak akan dimuat.
    RUN_OUT.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, object]] = []
    per_class: List[Dict[str, object]] = []
    confusion: List[Dict[str, object]] = []
    thresholds: List[Dict[str, object]] = []
    circuits: List[Dict[str, object]] = []
    inventory: List[Dict[str, object]] = []

    for track, workbook in WORKBOOKS:
        if not workbook.exists():
            print(f"PERINGATAN: {workbook.name} tidak ditemukan", flush=True)
            continue
        if phase not in ("all", "summary"):
            inventory.append({"track": track, "source": workbook.name, "run_count": 0, "per_class_rows": 0,
                              "confusion_rows": 0, "threshold_rows": 0})
            continue
        print(f"  Membaca {track}: {workbook.name}", flush=True)
        track_runs = [normalize_run(track, row) for row in iter_sheet(workbook, "01_Run_Summary")]
        runs.extend(track_runs)
        print(f"    ringkasan run: {len(track_runs)}", flush=True)

        class_rows = []
        for row in iter_sheet(workbook, "09_Per_Class_Metrics"):
            identity = normalize_identity(track, row)
            record = {**identity}
            for field in ["K", "feature_method", "seed", "class", "support", "precision", "recall", "f1", "correct", "wrong"]:
                record[field] = row.get(field)
            support = record.get("support") or 0
            record["class_accuracy"] = (record.get("correct") or 0) / support if support else None
            class_rows.append(record)
        per_class.extend(class_rows)
        print(f"    metrik per kelas: {len(class_rows)}", flush=True)

        confusion_rows = []
        for row in iter_sheet(workbook, "10_Confusion_Data"):
            identity = normalize_identity(track, row)
            confusion_rows.append({
                **identity,
                "K": row.get("K"),
                "feature_method": row.get("feature_method"),
                "seed": row.get("seed"),
                "actual_class": row.get("actual_class"),
                "predicted_class": row.get("predicted_class"),
                "count": row.get("count"),
            })
        confusion.extend(confusion_rows)
        print(f"    sel confusion matrix: {len(confusion_rows)}", flush=True)

        threshold_rows = []
        for row in iter_sheet(workbook, "19_Threshold_Sensitivity"):
            identity = normalize_identity(track, row)
            threshold_rows.append({
                **identity,
                "K": row.get("K"),
                "feature_method": row.get("feature_method"),
                "seed": row.get("seed"),
                "checkpoint": row.get("checkpoint"),
                "threshold": row.get("threshold"),
                "is_primary_threshold": row.get("is_primary_threshold"),
                "n_active": row.get("n_active"),
                "n_active_yodd": row.get("n_active_yodd"),
            })
        thresholds.extend(threshold_rows)
        print(f"    sensitivitas threshold: {len(threshold_rows)}", flush=True)

        if track == "PRIMARY":
            for row in iter_sheet(workbook, "18_Circuit_Benchmark"):
                circuits.append({
                    field: row.get(field) for field in [
                        "block_id", "model", "n_params", "gate_count", "depth",
                        "n_warmup", "n_repeat", "wall_median_sec", "wall_q1_sec",
                        "wall_q3_sec", "wall_p95_sec", "cpu_median_sec", "batch_size",
                        "unitary_median_sec", "device_name", "sim_dtype", "cpu_model",
                    ]
                })

        inventory.append({
            "track": track,
            "source": workbook.name,
            "run_count": len(track_runs),
            "per_class_rows": len(class_rows),
            "confusion_rows": len(confusion_rows),
            "threshold_rows": len(threshold_rows),
        })

    history_counts: Dict[str, Dict[str, int]] = {
        item["track"]: {"clustering_runs": item["run_count"], "supervised_runs": item["run_count"]} for item in inventory}
    if phase in ("all", "history"):
        print("  Menulis riwayat evaluasi lengkap per run...", flush=True)
    for track, workbook in WORKBOOKS:
        if phase not in ("all", "history") or (only_track and track != only_track) or not workbook.exists():
            continue
        c_count = s_count = 0
        if only_part in (None, "clustering"):
            print(f"    {track}: clustering", flush=True)
            c_count = extract_histories(track, workbook, "03_Clustering_History", CLUSTER_FIELDS, "clustering")
        if only_part in (None, "supervised"):
            print(f"    {track}: supervised", flush=True)
            s_count = extract_histories(track, workbook, "04_Supervised_History", SUPERVISED_FIELDS, "supervised")
        history_counts[track] = {"clustering_runs": c_count, "supervised_runs": s_count}

    if phase in ("all", "details"):
        print("  Menambahkan detail lengkap setiap run...", flush=True)
    for track, workbook in WORKBOOKS:
        if phase not in ("all", "details") or (only_track and track != only_track) or not workbook.exists():
            continue
        for sheet_name, part_name, fields in DETAIL_SHEETS:
            if only_part and part_name != only_part:
                continue
            print(f"    {track}: {part_name}", flush=True)
            extract_histories(track, workbook, sheet_name, fields, part_name)

    if phase not in ("all", "summary"):
        print(f"Fase {phase} selesai dalam {time.time() - started:.1f} detik.", flush=True)
        return 0
    stats = {
        "paired_primary": read_csv("paired_primary_D_minus_A.csv"),
        "ablation_contrasts": read_csv("ablation_contrasts.csv"),
        "more_comparison": read_csv("more_comparison.csv"),
        "mcnemar": read_csv("mcnemar_primary.csv"),
        "six_nine": read_csv("g1_11_six_nine.csv"),
        "baselines": read_csv("baseline_comparison.csv"),
    }
    aggregates = read_csv("seed_aggregation.csv")

    completed = sum(1 for run in runs if run.get("status") == "COMPLETED")
    payload = {
        "metadata": {
            "title": "MORE-HD Training Observatory — MADBase",
            "dataset": "MADBASE-AUC-V1",
            "glyphs": "٠١٢٣٤٥٦٧٨٩",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_root": "research_arabic_digits/research_data",
            "run_count": len(runs),
            "completed_run_count": completed,
            "tracks": sorted({str(run.get("track")) for run in runs}),
            "models": sorted({str(run.get("architecture")) for run in runs}),
            "features": sorted({str(run.get("feature_method")) for run in runs}),
            "k_values": sorted({int(run["K"]) for run in runs if isinstance(run.get("K"), int)}),
            "history_counts": history_counts,
            "detail_sections": [part_name for _sheet, part_name, _fields in DETAIL_SHEETS],
        },
        "runs": runs,
        "aggregates": aggregates,
        "per_class": per_class,
        "confusion": confusion,
        "thresholds": thresholds,
        "circuits": circuits,
        "stats": stats,
        "inventory": inventory,
        "findings": read_findings(),
    }
    write_json(OUT / "dashboard-data.json", payload)
    write_json(OUT / "manifest.json", payload["metadata"], pretty=True)

    elapsed = time.time() - started
    size_mb = (OUT / "dashboard-data.json").stat().st_size / 1024 / 1024
    print(
        f"Selesai: {len(runs)} run, {len(per_class)} baris per-kelas, "
        f"{len(confusion)} sel confusion, dashboard-data {size_mb:.1f} MB "
        f"dalam {elapsed:.1f} detik.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Bangun data dashboard MADBase. Tanpa argumen = semua fase.")
    _ap.add_argument("--phase", choices=["all", "summary", "history", "details"], default="all")
    _ap.add_argument("--track", choices=["PRIMARY", "ABLATION", "MORE_REFERENCE"], default=None)
    _ap.add_argument("--part", default=None, help="history: clustering/supervised; details: config/correlation/...")
    _args = _ap.parse_args()
    raise SystemExit(main(_args.phase, _args.track, _args.part))
