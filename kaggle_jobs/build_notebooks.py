"""Build the Kaggle runner notebooks. The code is embedded as a base64 tarball, so every pushed version is
self-contained (no git clone needed) and exactly reproducible.

python kaggle_jobs/build_notebooks.py --role task1 [--smoke] [--resume-from khizaryy/<slug>] [--pool49]
writes kaggle_deploy/<slug>/{runner.ipynb,kernel-metadata.json}; push with
kaggle kernels push -p kaggle_deploy/<slug> --accelerator NvidiaTeslaT4
"""
import argparse
import base64
import io
import json
import os
import tarfile

USER = "khizaryy"
CODE = ["common", "task1_ft", "task2_vlm", "task3_ttc", "kaggle_jobs", "data/task1", "data/task3", "requirements.txt"]
DEADLINE_H = 11.25  # Kaggle hard limit is 12 h; leave time for the output to be saved


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for p in CODE:
            t.add(p, filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
    return base64.b64encode(buf.getvalue()).decode()


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


def build(role, smoke, slug, resume_from, pool49):
    title = {"task1": "Task 1 (LLM fine-tuning)", "task23": "Task 2 (VLM) + Task 3 live generation"}[role]
    b64 = bundle()
    cells = [
        cell(f"# ATML PA3 runner: {title}{' [SMOKE]' if smoke else ''}\n"
             "Runs headless on GPU T4 x2: one job queue per GPU. Resumable: attach this notebook's previous output as an input and rerun; "
             "finished jobs are skipped and interrupted trainings resume from their checkpoints.", "markdown"),
        cell(f"""import os, time, sys, subprocess, glob, shutil
T0 = time.time()
ROLE, SMOKE, POOL49 = {role!r}, {smoke!r}, {pool49!r}
DEADLINE = T0 + {DEADLINE_H if not smoke else 1.5} * 3600
WORK = '/kaggle/working/pa3'
os.makedirs(WORK, exist_ok=True)
os.environ.update(PA3_DEADLINE=str(DEADLINE), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
                  HF_HOME='/tmp/hf', PA3_SCRATCH='/tmp/pa3_scratch', PA3_RAW='/tmp/pa3_raw', PA3_IMG='/tmp/coco_img',
                  HF_HUB_DISABLE_PROGRESS_BARS='1', TRANSFORMERS_VERBOSITY='warning')
print('role', ROLE, 'smoke', SMOKE, 'deadline in', (DEADLINE - T0) / 3600, 'h')"""),
        cell(f"""# unpack the embedded code (only code/data files are overwritten; outputs/results/state are kept)
import base64, io, tarfile
CODE_B64 = {b64!r}
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode='r:gz').extractall(WORK)
os.chdir(WORK)
print(sorted(os.listdir(WORK)))"""),
        cell("""# restore a previous session's progress if its output is attached as an input
prev = sorted(glob.glob('/kaggle/input/**/pa3/state/status.json', recursive=True))
for st in prev:
    src = os.path.dirname(os.path.dirname(st))
    print('restoring from', src)
    for sub in ['outputs', 'results', 'state', 'logs']:
        if os.path.exists(os.path.join(src, sub)):
            subprocess.call(f'cp -rn {src}/{sub} {WORK}/', shell=True)
print('done markers:', sorted(os.listdir('state/done')) if os.path.exists('state/done') else 'none')"""),
        cell("""# environment (Kaggle image torch/transformers are used as-is)
subprocess.call('pip install -q "lm-eval==0.4.13" "math-verify==0.9.0" 2>&1 | tail -3', shell=True)
subprocess.call('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv', shell=True)
subprocess.call('python -c "from common.utils import env_info; import json; print(json.dumps(env_info(), indent=1))"', shell=True)
subprocess.call('df -h /kaggle/working /tmp | cat; free -g', shell=True)"""),
        cell("""# run the job queues (streams progress; full per-job logs in logs/<job>.log)
cmd = f'python -m kaggle_jobs.run --role {ROLE}' + (' --smoke' if SMOKE else '') + (' --pool49' if POOL49 else '')
rc = subprocess.call(cmd, shell=True)
print('orchestrator exit code', rc)"""),
        cell("""# summary
import json
print(open('state/status.json').read() if os.path.exists('state/status.json') else 'no status')
subprocess.call('du -sh outputs/* results/* 2>/dev/null; du -sh /kaggle/working', shell=True)
print('session hours', (time.time() - T0) / 3600)"""),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    d = f"kaggle_deploy/{slug}"
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(f"{d}/runner.ipynb", "w"), indent=1)
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": "runner.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": ["jeffaudi/coco-2014-dataset-for-yolov3"] if role == "task23" else [],
            "competition_sources": [], "kernel_sources": [resume_from] if resume_from else [], "model_sources": []}
    json.dump(meta, open(f"{d}/kernel-metadata.json", "w"), indent=2)
    print(f"built {d}: {os.path.getsize(f'{d}/runner.ipynb') / 1024:.0f} KB, meta {meta}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["task1", "task23"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--pool49", action="store_true")
    a = ap.parse_args()
    slug = a.slug or f"atml-pa3-{a.role}{'-smoke' if a.smoke else ''}"
    build(a.role, a.smoke, slug, a.resume_from, a.pool49)
