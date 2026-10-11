"""Run the frozen MADBase study, then export and analyse it; preserve completed results."""
import argparse
import ctypes
import json
import msvcrt
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import constants as C
for variable in C.THREAD_ENV_VARS:
    os.environ[variable] = "1"
from core.artifact_io import load_json, save_json_atomic
from core.classical_baselines import baseline_config, write_results_csv
from core.config import RunConfig, validate_config
from core.run_status import existing_attempts, read_status
from scripts.confirmatory_worker import baseline_directory, completed_source
from scripts.print_run_plan import entries

STUDY = ROOT / "research_arabic_digits/research_data"
QUEUE = STUDY / "confirmatory_machine/queue"


def parallel_limit(default):
    control = QUEUE / "control.json"
    value = load_json(control).get("parallel", default) if control.exists() else default
    if type(value) is not int or not 1 <= value <= 8:
        raise ValueError("Queue parallel limit must be an integer in 1..8")
    return value


class AdoptedProcess:
    """Observe an existing Windows training process without restarting it."""
    def __init__(self, pid):
        self.pid = pid
        self.exit_code = None
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        self.kernel.OpenProcess.restype = ctypes.c_void_p
        self.kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        self.handle = self.kernel.OpenProcess(0x101000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def poll(self):
        if self.exit_code is not None:
            return self.exit_code
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result == 258:
            return None
        if result != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = ctypes.c_uint32()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.exit_code = code.value
        self.kernel.CloseHandle(self.handle)
        return self.exit_code


def main_tasks(parallel):
    decision = load_json(STUDY / "pilot/convergence_budget_decision.json")
    if decision["status"] != "FROZEN" or decision["dataset"] != "MADBASE" or decision["n_runs"] != 14:
        raise RuntimeError("MADBase budget is not frozen from the full pilot")
    tasks = []
    for track, script, flag, arch, feature, k, seed in entries():
        cfg = RunConfig(arch, feature, k, seed, dataset="MADBASE", run_mode="CONFIRMATORY",
                        project_root=str(ROOT), n_parallel_declared=parallel,
                        max_nfev_clustering=decision["clustering"]["final_max_nfev"],
                        max_nfev_supervised=decision["supervised"]["final_max_nfev"])
        validate_config(cfg)
        tasks.append(dict(key=cfg.run_uid, kind="main", cfg=cfg, script=script, flag=flag))
    if len(tasks) != 450 or len({t["key"] for t in tasks}) != 450:
        raise RuntimeError("Invalid confirmatory plan")
    return tasks


def main_command(task):
    cfg = task["cfg"]
    attempts = existing_attempts(cfg.run_dir)
    completed = [path for _, path, st in attempts if st and st.get("status") == "COMPLETED"]
    if len(completed) > 1:
        raise RuntimeError(f"Duplicate completed identity: {cfg.run_uid}")
    if completed:
        directory = completed[0]
        if (directory / "workbook_export_complete.json").exists():
            return None, directory
        return [sys.executable, "-u", str(ROOT / "scripts/confirmatory_worker.py"), "export",
                "--run-dir", str(directory)], directory
    if any(not st or st.get("status") != "FAILED" for _, _, st in attempts):
        raise RuntimeError(f"Existing unfinished attempt needs review: {cfg.run_uid}")
    attempt = max((n for n, _, _ in attempts), default=0) + 1
    cfg.attempt = attempt
    cmd = [sys.executable, "-u", str(ROOT / task["script"])] + task["flag"].split()
    cmd += ["--dataset", "MADBASE", "--feature", cfg.feature_method, "--k", str(cfg.k),
            "--seed", str(cfg.seed), "--run-mode", "CONFIRMATORY", "--n-parallel-declared",
            str(cfg.n_parallel_declared), "--project-root", str(ROOT), "--attempt", str(attempt)]
    return cmd, cfg.run_dir


def worker_task(kind, architecture, feature, k, seed, parallel):
    key = f"{kind}_{architecture}_{feature}_K{k}_S{seed}"
    cmd = [sys.executable, "-u", str(ROOT / "scripts/confirmatory_worker.py"), kind,
           "--architecture", architecture, "--feature", feature, "--k", str(k), "--seed", str(seed),
           "--parallel", str(parallel)]
    return dict(key=key, kind=kind, command=cmd, seed=seed, feature=feature, k=k, architecture=architecture)


def verify_finished(task):
    if task["kind"] == "main":
        directory = task["directory"]
        status = read_status(directory)
        cfg = load_json(directory / "config.json")
        if not status or status["status"] != "COMPLETED" or cfg["run_uid"] != task["key"]:
            raise RuntimeError(f"Missing completed artifacts: {task['key']}")
        workbook = directory / C.LOCAL_RUN_SPREADSHEET_NAME
        if not workbook.is_file() or workbook.stat().st_size == 0:
            raise RuntimeError(f"Missing per-run Excel: {task['key']}")
        save_json_atomic({"run_uid": task["key"], "completed_at": datetime.now().astimezone().isoformat()},
                         directory / "workbook_export_complete.json")
    elif task["kind"] == "baseline":
        rows = load_json(baseline_directory(task["seed"], task["feature"], task["k"]) / "baseline_rows.json")
        if len(rows) != 3:
            raise RuntimeError("Incomplete baseline export")


def run_phase(name, tasks, parallel, state, adoption=None):
    pending, active = list(tasks), {}
    done, failures = [], []
    if adoption and adoption.get("phase") == name:
        lookup = {task["key"]: task for task in pending}
        for previous in adoption["active"]:
            task = lookup[previous["task"]]
            if task["kind"] != "main":
                raise RuntimeError("Only main training processes can be adopted")
            attempts = existing_attempts(task["cfg"].run_dir)
            if not attempts:
                raise RuntimeError(f"No existing attempt to adopt: {task['key']}")
            directory = attempts[-1][1]
            try:
                process = AdoptedProcess(previous["pid"])
            except OSError:
                # A child can finish in the short launcher handover; export from its saved model.
                status = read_status(directory)
                if status and status.get("status") == "COMPLETED":
                    continue
                raise
            task.update(directory=directory, process=process, log=None, log_path=previous["log"])
            active[process.pid] = task
            pending.remove(task)
            print(f"ADOPT {task['key']} PID={process.pid}", flush=True)
    # Count results already exported before a queue restart without training them again.
    for task in pending[:]:
        if task["kind"] == "main":
            attempts = existing_attempts(task["cfg"].run_dir)
            complete = [directory for _, directory, st in attempts if st and st.get("status") == "COMPLETED"]
            if len(complete) > 1:
                raise RuntimeError(f"Duplicate completed identity: {task['key']}")
            if complete and (complete[0] / "workbook_export_complete.json").exists():
                done.append(task["key"])
                pending.remove(task)
    state.update(phase=name, phase_total=len(tasks), status="RUNNING")
    while pending or active:
        limit = parallel_limit(parallel)
        state["parallel"] = limit
        for pid, task in list(active.items()):
            rc = task["process"].poll()
            if rc is None:
                continue
            if task["log"] is not None:
                task["log"].close()
            try:
                if rc != 0:
                    raise RuntimeError(f"Process exited {rc}; see {task['log_path']}")
                verify_finished(task)
                done.append(task["key"])
                print(f"DONE [{len(done)}/{len(tasks)}] {task['key']}", flush=True)
            except Exception as exc:
                failures.append(dict(task=task["key"], error=str(exc)))
                print(f"FAILED {task['key']}: {exc}", flush=True)
            del active[pid]
        available_disk = shutil.disk_usage(ROOT).free
        state["resource_hold"] = available_disk < 1024 ** 3
        while pending and len(active) < limit and not failures and not state["resource_hold"]:
            task = pending.pop(0)
            if task["kind"] == "main":
                task["cfg"].n_parallel_declared = limit
                cmd, directory = main_command(task)
                task["directory"] = directory
                if cmd is None:
                    done.append(task["key"])
                    continue
            else:
                cmd = list(task["command"])
                cmd[cmd.index("--parallel") + 1] = str(limit)
            log_path = QUEUE / "logs" / f"{task['key']}_{time.time_ns()}.log"
            log = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            task.update(process=process, log=log, log_path=str(log_path))
            active[process.pid] = task
            print(f"START {task['key']} PID={process.pid}", flush=True)
        state.update(updated_at=datetime.now().astimezone().isoformat(), phase_completed=len(done),
                     completed_keys=done, pending_count=len(pending),
                     active=[dict(task=t["key"], pid=pid, log=t["log_path"]) for pid, t in active.items()],
                     failures=failures, free_disk_gb=round(available_disk / 1024**3, 2))
        save_json_atomic(state, QUEUE / "status.json")
        if failures and not active:
            state["status"] = "NEEDS_ATTENTION"
            save_json_atomic(state, QUEUE / "status.json")
            raise RuntimeError(f"{name} requires review; completed results retained")
        if pending or active:
            time.sleep(3)
    state["completed_phases"][name] = len(done)
    save_json_atomic(state, QUEUE / "status.json")


def postprocess(command, label, state):
    state.update(phase=label, active=[], pending_count=0)
    save_json_atomic(state, QUEUE / "status.json")
    path = QUEUE / "logs" / f"{label}_{time.time_ns()}.log"
    with path.open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--adopt-running", type=Path, help="Previous queue snapshot for preserving live training")
    args = parser.parse_args()
    if not 1 <= args.parallel <= 8:
        parser.error("parallel must be 1..8")
    tasks = main_tasks(args.parallel)
    if args.dry_run:
        print(json.dumps(dict(main=len(tasks), tracks=dict(Counter(t['cfg'].track for t in tasks)),
                              secondary_runs=90, baseline_cells=120, baseline_fits=240,
                              parallel=args.parallel, per_run_workbook=True, auto_consolidation=True,
                              auto_analysis=True), indent=2))
        return
    QUEUE.mkdir(parents=True, exist_ok=True)
    (QUEUE / "logs").mkdir(exist_ok=True)
    lock = (QUEUE / "exclusive.lock").open("a+b")
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    state = dict(started_at=datetime.now().astimezone().isoformat(), pid=os.getpid(), parallel=args.parallel,
                 dataset="MADBASE", status="RUNNING", completed_phases={}, main_total=450,
                 secondary_total=90, baseline_cells_total=120)
    adoption = load_json(args.adopt_running) if args.adopt_running else None
    if adoption:
        if adoption.get("dataset") != "MADBASE" or adoption.get("phase") != "MAIN":
            raise RuntimeError("Queue adoption snapshot does not match MADBase MAIN")
        state["started_at"] = adoption["started_at"]
        state["completed_phases"] = adoption.get("completed_phases", {})
        state["parallel_change"] = dict(previous=adoption["parallel"], requested=args.parallel,
                                       changed_at=datetime.now().astimezone().isoformat(),
                                       policy="Preserve running work; start new runs only below the requested limit")
    try:
        audit = load_json(STUDY / "manifests/readiness_report.json")
        if not audit.get("confirmatory_ready"):
            raise RuntimeError("MADBase readiness gate is not open")
        run_phase("MAIN", tasks, args.parallel, state, adoption)
        secondary = [worker_task("jalurb", t["cfg"].architecture, t["cfg"].feature_method,
                     t["cfg"].k, t["cfg"].seed, args.parallel) for t in tasks
                     if t["cfg"].track == "PRIMARY" and t["cfg"].k in C.JALUR_B_K_VALUES]
        if len(secondary) != 90:
            raise RuntimeError("Wrong secondary matrix")
        run_phase("JALUR_B", secondary, args.parallel, state)
        cells = [worker_task("baseline", "MORE-HD", feature, k, seed, args.parallel)
                 for seed in C.CONFIRMATORY_SEEDS for k in C.K_VALUES for feature in C.FEATURE_METHODS]
        run_phase("BASELINES", cells, args.parallel, state)
        baseline_root = ROOT / "runs/arabic_digits/baselines"
        rows = []
        for task in cells:
            rows.extend(load_json(baseline_directory(task["seed"], task["feature"], task["k"]) / "baseline_rows.json"))
        if len(rows) != 360 or sum(r["baseline"] in ("NC", "LR") for r in rows) != 240:
            raise RuntimeError("Incomplete classical matrix")
        write_results_csv(rows, baseline_root / "baseline_results.csv")
        save_json_atomic({**baseline_config(), "dataset":"MADBASE", "dataset_id":C.DATASET_IDS["MADBASE"],
                          "run_mode":"CONFIRMATORY", "seeds":list(C.CONFIRMATORY_SEEDS),
                          "features":list(C.FEATURE_METHODS), "K_values":list(C.K_VALUES)},
                         baseline_root / "baseline_config.json")
        save_json_atomic([], baseline_root / "baseline_errors.json")
        benchmarks = STUDY / "confirmatory_machine/benchmarks"
        if not all((benchmarks / f"circuit_microbenchmark_block{b}.json").exists() for b in (1, 2, 3)):
            postprocess(["scripts/circuit_microbenchmark.py", "--out-dir", str(benchmarks)], "CIRCUIT_BENCHMARK", state)
        postprocess(["scripts/verify_confirmatory_completion.py"], "FINAL_ADMISSION", state)
        postprocess(["scripts/consolidate_runs.py", "--dataset", "MADBASE", "--run-mode", "CONFIRMATORY"], "CONSOLIDATE_EXCEL", state)
        postprocess(["analysis/run_analysis.py", "--dataset", "MADBASE", "--run-mode", "CONFIRMATORY"], "STATISTICAL_ANALYSIS", state)
        state.update(status="COMPLETED", phase="FINISHED", finished_at=datetime.now().astimezone().isoformat())
        save_json_atomic(state, QUEUE / "status.json")
        print("FULL MADBASE STUDY COMPLETED: 450 main + 90 secondary + 240 classical fits; Excel and analysis saved", flush=True)
    except BaseException as exc:
        state.update(status="NEEDS_ATTENTION", error=str(exc), updated_at=datetime.now().astimezone().isoformat())
        save_json_atomic(state, QUEUE / "status.json")
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    main()
