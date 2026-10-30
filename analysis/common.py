"""Shared plotting style for report figures (static PDFs).

Categorical slots follow the validated reference palette order; each method keeps its colour in every figure.
Three slots are below 3:1 contrast on white, so every line also gets a distinct style/marker and figures are
paired with tables (relief rule).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = "results/figures"
TAB = "results/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
T1 = {"fullft": ("Full FT", SLOTS[0], "-", "o"), "l2sp": ("Full FT + L2-SP", SLOTS[1], "--", "s"), "lora4": ("LoRA-4", SLOTS[2], "-.", "^"),
      "lora16": ("LoRA-16", SLOTS[3], ":", "D"), "oplora": ("OPLoRA-16", SLOTS[4], (0, (5, 1, 1, 1)), "v")}
T1_COND = {"baseline": "Baseline", "fullft_mid": "Full FT (mid)", "fullft": "Full FT", "l2sp": "Full FT + L2-SP",
           "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora": "OPLoRA-16"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def jl(p):
    return [json.loads(l) for l in open(p)]


def jd(p):
    return json.load(open(p))


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png", dpi=200)
    plt.close(fig)


def write_table(rows, cols, name, fmt=None):
    """csv + a booktabs latex body (header + rows) for the report"""
    fmt = fmt or {}
    with open(f"{TAB}/{name}.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    def cell(c, v):
        if v is None or v == "":
            return "--"
        return fmt[c](v) if c in fmt else str(v)

    with open(f"{TAB}/{name}.tex", "w") as f:
        f.write("\\toprule\n" + " & ".join(cols) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(cell(c, r.get(c)) for c in cols) + " \\\\\n")
        f.write("\\bottomrule\n")
