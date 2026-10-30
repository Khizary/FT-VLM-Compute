"""Kaggle orchestrator: one job queue per T4, dependency-aware, resumable, deadline-aware.

python -m kaggle_jobs.run --role task1|task23 [--smoke]
Each job is a subprocess pinned to one GPU. A finished job writes state/done/<job>.json, so a rerun of the
notebook (with the previous output attached) skips it; interrupted trainings resume from their checkpoints.
Training scripts exit 75 when they stop for the deadline; that job is left 'pending' for the next session.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

STATE = "state"
LOGS = "logs"
H = 3600


def jobs_task1(smoke):
    tr = lambda r, extra="": f"python -m task1_ft.train --run {r} {extra}"
    if smoke:
        small = "--max-updates 2 --ckpt-at 1,2 --mid 1"
        g0 = [("train_fullft_a", tr("fullft", "--max-updates 4 --ckpt-at 2,4 --mid 2 --stop-after 2"), []),
              ("train_fullft", tr("fullft", "--max-updates 4 --ckpt-at 2,4 --mid 2"), []),  # resumes from _a's preemption ckpt
              ("train_l2sp", tr("l2sp", small), []),
              ("eval_baseline", "python -m task1_ft.evaluate --cond baseline --limit 8", []),
              ("eval_fullft_mid", "python -m task1_ft.evaluate --cond fullft_mid --limit 4", ["train_fullft"]),
              ("eval_l2sp", "python -m task1_ft.evaluate --cond l2sp --limit 4", ["train_l2sp"])]
        g1 = [("train_lora16", tr("lora16", "--max-updates 4 --ckpt-at 2,4 --mid 2"), []),
              ("train_oplora", tr("oplora", small), []),
              ("train_lora4", tr("lora4", small), []),
              ("eval_oplora", "python -m task1_ft.evaluate --cond oplora --limit 4", ["train_oplora"]),
              ("eval_lora4", "python -m task1_ft.evaluate --cond lora4 --limit 4", ["train_lora4"]),
              ("diagnostics", "python -m task1_ft.diagnostics", ["train_fullft", "train_l2sp", "train_lora4", "train_lora16", "train_oplora"])]
        return g0, g1
    ev = lambda c, deps: (f"eval_{c}", f"python -m task1_ft.evaluate --cond {c}", deps)
    # balanced from smoke timings (s/update: full 6.3, lora16 8.0, lora4 7.5, oplora 9.9; eval ~45 min): ~9.3 h per GPU
    g0 = [("train_fullft", tr("fullft"), []), ("train_l2sp", tr("l2sp"), []), ("train_oplora", tr("oplora"), []),
          ev("oplora", ["train_oplora"]), ev("baseline", [])]
    g1 = [("train_lora16", tr("lora16"), []), ("train_lora4", tr("lora4"), []),
          ev("fullft_mid", ["train_fullft"]), ev("fullft", ["train_fullft"]), ev("l2sp", ["train_l2sp"]),
          ev("lora16", ["train_lora16"]), ev("lora4", ["train_lora4"]),
          ("diagnostics", "python -m task1_ft.diagnostics", ["train_fullft", "train_l2sp", "train_lora4", "train_lora16", "train_oplora"])]
    return g0, g1


def jobs_task23(smoke, pool49):
    sm = " --smoke" if smoke else ""
    g0 = [("t2_cache", "python -m task2_vlm.cache_features", ["t2_prepare"]),
          ("t2_train_main", "python -m task2_vlm.train --run main" + sm, ["t2_cache"]),
          ("t2_eval_main", "python -m task2_vlm.evaluate --run main", ["t2_train_main"])]
    g1 = [("t3_live", "python -m task3_ttc.live_generate" + (" --limit 1" if smoke else ""), [])]
    if pool49:
        g1 += [("t2_train_pool49", "python -m task2_vlm.train --run pool49 --pool" + sm, ["t2_cache"]),
               ("t2_eval_pool49", "python -m task2_vlm.evaluate --run pool49 --pool", ["t2_train_pool49"])]
    return g0, g1


class Runner:
    def __init__(self, deadline):
        self.deadline = deadline
        self.lock = threading.Lock()
        self.status = {}
        os.makedirs(f"{STATE}/done", exist_ok=True)
        os.makedirs(LOGS, exist_ok=True)

    def done(self, j):
        return os.path.exists(f"{STATE}/done/{j}.json")

    def say(self, msg):
        with self.lock:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def save_status(self):
        with self.lock:
            json.dump({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": self.status, "remaining_h": (self.deadline - time.time()) / H},
                      open(f"{STATE}/status.json", "w"), indent=2)

    def run_job(self, name, cmd, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        t0 = time.time()
        self.status[name] = {"state": "running", "gpu": gpu, "cmd": cmd, "start": time.strftime("%H:%M:%S")}
        self.save_status()
        self.say(f">>> [gpu{gpu}] {name}: {cmd}")
        with open(f"{LOGS}/{name}.log", "a") as lf:
            p = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                lf.write(line)
                lf.flush()
                # keep the notebook log readable: forward our own log lines and errors, not tqdm bars
                if line.startswith("[") or "Error" in line or "Traceback" in line or "error" in line.lower()[:40]:
                    self.say(f"[{name}] {line.rstrip()[:400]}")
            rc = p.wait()
        dt = time.time() - t0
        st = "done" if rc == 0 else "preempted" if rc == 75 else "failed"
        self.status[name].update(state=st, rc=rc, minutes=round(dt / 60, 1))
        if rc == 0:
            json.dump({"job": name, "minutes": dt / 60, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, open(f"{STATE}/done/{name}.json", "w"))
        self.save_status()
        self.say(f"<<< [gpu{gpu}] {name}: {st} rc={rc} after {dt / 60:.1f} min")
        if rc not in (0, 75):
            os.system(f"tail -n 40 {LOGS}/{name}.log")
        return st

    def worker(self, gpu, queue):
        for name, cmd, deps in queue:
            if self.done(name):
                self.say(f"skip {name} (done in an earlier session)")
                self.status[name] = {"state": "done-earlier"}
                continue
            while not all(self.done(d) for d in deps):
                if any(self.status.get(d, {}).get("state") in ("failed", "preempted", "skipped") for d in deps):
                    break
                if time.time() > self.deadline:
                    break
                time.sleep(30)
            if not all(self.done(d) for d in deps):
                self.status[name] = {"state": "skipped", "reason": "dependency not finished"}
                self.say(f"skip {name}: dependency not finished")
                continue
            left = self.deadline - time.time()
            if not name.startswith(("train", "t2_train", "t3_live")) and left < 50 * 60:
                self.status[name] = {"state": "skipped", "reason": f"only {left / 60:.0f} min left"}
                self.say(f"skip {name}: only {left / 60:.0f} min left")
                continue
            if left < 15 * 60:
                self.status[name] = {"state": "skipped", "reason": "deadline"}
                continue
            self.run_job(name, cmd, gpu)
        self.save_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["task1", "task23"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pool49", action="store_true", help="also run the optional 49-token Task 2 experiment")
    a = ap.parse_args()
    deadline = float(os.environ["PA3_DEADLINE"])
    r = Runner(deadline)
    r.say(f"role={a.role} smoke={a.smoke} deadline in {(deadline - time.time()) / H:.2f} h")

    if a.role == "task1":
        if not r.done("prep"):
            prefetch = ("python -c \"from datasets import load_dataset as L; L('openai/gsm8k','main'); L('Rowan/hellaswag'); "
                        "L('aps/super_glue','boolq'); L('allenai/ai2_arc','ARC-Easy'); "
                        "from huggingface_hub import snapshot_download as S; S('Qwen/Qwen3-0.6B')\"")
            if r.run_job("prefetch", prefetch, 0) != "done" or r.run_job("prep", "python -m task1_ft.data", 0) != "done":
                sys.exit(1)
        g0, g1 = jobs_task1(a.smoke)
    else:
        if not r.done("t2_prepare"):
            if r.run_job("t2_prepare", "python -m task2_vlm.prepare_data", 0) != "done":
                sys.exit(1)
        g0, g1 = jobs_task23(a.smoke, a.pool49)

    th = [threading.Thread(target=r.worker, args=(0, g0)), threading.Thread(target=r.worker, args=(1, g1))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    r.save_status()
    bad = {k: v for k, v in r.status.items() if v.get("state") not in ("done", "done-earlier")}
    r.say(f"ALL QUEUES FINISHED. not done: {json.dumps(bad, indent=1) if bad else 'none'}")


if __name__ == "__main__":
    main()
