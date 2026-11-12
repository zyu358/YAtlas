"""Leave-one-out CRF-ID on the 23 inx-4p & mbr-1p animals of Fig. 3a.

Each animal is held out in turn. Its nuclei are identified against the
data-driven atlas built from the other 22 animals, or against a reference atlas
restricted to the identities those 22 animals cover.

  compare  baseline vs edge-trust CRF-ID on the data-driven atlas, every term
           weight 1 by default (--weights AP LR DV DIST ANGLE to change).
  fig3a    the Fig. 3a values: the data-driven atlas with the Fig. 3a
           edge-trust recipe, and the NeuroPAL and OpenWorm reference atlases
           with the reference recipe; compared with the stored values in
           data/fig3a_inx4_expected.csv.

Usage (from the repository root):
    python demo/run_demo.py [--part all|compare|fig3a] [--weights 1 1 1 1 1] [--workers 4]
Tables and figures are written to demo/results/.
"""

import argparse
import functools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from crfid import Atlas, Image, baseline, edge_trust, is_labeled  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
THRESHOLD = 5                    # a candidate is labelled in >= 5 atlas animals
ATLAS_XYZ = ["atlas_ap_um", "atlas_lr_um", "atlas_dv_um"]
HELDOUT_XYZ = ["heldout_ap_um", "heldout_lr_um", "heldout_dv_um"]
XYZ = ["ap_um", "lr_um", "dv_um"]

# Fig. 3a recipes. Weights: AP, LR, DV order, distance, angle.
OURS_CFG = edge_trust.Config(weights=(1.0, 1.0, 1.0, 0.2, 5.0), unary_weight=1.0, edge_trust=True,
                             trust_floor=0.5, message_trust=True, knn=10, max_iters=50)
REF_CFG = edge_trust.Config(weights=(1.0, 1.0, 1.0, 1.0, 0.5), unary_weight=2.0, edge_trust=False,
                            message_trust=False, knn=10, max_iters=50)

STRAIN_PARTS = (("inx-4p", "italic"), ("&", "normal"), ("mbr-1p", "italic"))
BLUE, ORANGE, RED, GRAY, INK = "#3b6ea5", "#d98c4a", "#c0504d", "#9aa0a6", "#1a1a1a"


def read(name):
    """Read a demo table; floats are parsed exactly and labels such as 'NA' stay strings."""
    return pd.read_csv(DATA / name, keep_default_na=False, float_precision="round_trip")


@functools.lru_cache(maxsize=None)
def load():
    """Detections, per-fold candidates, scored identities and reference atlases."""
    det = read("inx4_detections.csv")
    worms = sorted(det.worm.unique())
    lab = det[det.label.map(is_labeled)]
    atlas_images = {w: Image(w, g[ATLAS_XYZ].to_numpy(float), g.label)
                    for w, g in lab.sort_values(["worm", "label"]).groupby("worm")}
    counts = lab.groupby("label").worm.nunique()
    scored = set(counts[counts >= THRESHOLD].index)
    cand = read("inx4_candidates.csv")
    candidates = {w: g.label.tolist() for w, g in cand.groupby("worm")}
    for w in worms:
        c = lab[lab.worm != w].groupby("label").worm.nunique()
        assert candidates[w] == sorted(c[c >= THRESHOLD].index), f"{w}: candidate list differs"
    npal = read("inx4_neuropal_atlas.csv")
    neuropal = [Image(s, g[XYZ].to_numpy(float), g.label) for s, g in npal.groupby("specimen")]
    ow = read("inx4_openworm_atlas.csv")
    openworm = Image("OpenWorm", ow[XYZ].to_numpy(float), ow.label)
    return dict(det=det, worms=worms, atlas_images=atlas_images, scored=scored, candidates=candidates,
                neuropal=neuropal, openworm=openworm)


def accuracy(labels, pred, scored):
    """Fraction of annotated nuclei with an identity in `scored` whose prediction is correct."""
    hits = [g == p for g, p in zip(labels, pred) if is_labeled(g) and g in scored]
    return sum(hits) / len(hits)


def median_isotropic_sd(atlas):
    """Median over identities of sqrt(mean of the three per-axis variances)."""
    sd = atlas.pos_sd[np.isfinite(atlas.pos_sd).all(axis=1)]
    return float(np.median(np.sqrt((sd ** 2).mean(axis=1))))


def run_fold(k, parts, weights):
    """Hold out worm k: build the leave-one-out atlases and identify its nuclei."""
    D = load()
    worm, scored = D["worms"][k], D["scored"]
    atlas = Atlas([D["atlas_images"][w] for w in D["worms"] if w != worm], D["candidates"][worm], THRESHOLD)
    rows = D["det"][D["det"].worm == worm]
    fed = rows[rows.rule_feed == 1]
    target = Image(worm, fed[HELDOUT_XYZ].to_numpy(float), fed.label)       # data-driven atlas: rule feed
    out = dict(worm=worm, n_scored=sum(is_labeled(l) and l in scored for l in target.labels))

    if "compare" in parts:
        pred, _ = baseline.run_crf(atlas, target, weights=weights)
        out["baseline"] = accuracy(target.labels, pred, scored)
        pred, _ = edge_trust.run_crf(atlas, target, edge_trust.Config(weights=weights))
        out["edge_trust"] = accuracy(target.labels, pred, scored)
    if "fig3a" in parts:
        pred, _ = edge_trust.run_crf(atlas, target, OURS_CFG)
        out["ours"] = accuracy(target.labels, pred, scored)
        every = Image(worm, rows[HELDOUT_XYZ].to_numpy(float), rows.label)   # references: every detection
        at_np = Atlas(D["neuropal"], D["candidates"][worm], THRESHOLD)
        pred, _ = edge_trust.run_crf(at_np, every, REF_CFG)
        out["neuropal"] = accuracy(every.labels, pred, scored)
        ow_labels = set(D["openworm"].labels)
        at_ow = Atlas([D["openworm"]], [c for c in D["candidates"][worm] if c in ow_labels], threshold=1)
        at_ow.pos_sd = np.full_like(at_ow.mean_pos, median_isotropic_sd(at_np))   # one animal: no SD of its own
        pred, _ = edge_trust.run_crf(at_ow, every, REF_CFG)
        out["openworm"] = accuracy(every.labels, pred, scored)
    return out


# --------------------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if any(f.name == "Arial" for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
    return plt


def _header(fig, ax, lines, top=0.975):
    """Strain name (gene names in italics) and further lines, centred above the axes."""
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height

    def width(t):
        return t.get_window_extent(r).width

    parts = [fig.text(0, top, s, va="top", fontstyle=st) for s, st in STRAIN_PARTS]
    probe = [fig.text(0, 0, s) for s in ("x x", "xx")]
    space = width(probe[0]) - width(probe[1])
    for t in probe:
        t.remove()
    widths = [width(t) for t in parts]
    pos = ax.get_position()
    x = 0.5 * (pos.x0 + pos.x1) * W - (sum(widths) + space * (len(parts) - 1)) / 2
    for t, w in zip(parts, widths):
        t.set_x(x / W)
        x += w + space
    y = top - parts[0].get_window_extent(r).height / H - 0.012
    fig.text(0.5 * (pos.x0 + pos.x1), y, "\n".join(lines), ha="center", va="top", linespacing=1.4)


def paired_panel(ax, columns, names, colors):
    """Box plots (light fill, coloured edge), one dot per worm, grey lines joining the same worm."""
    vals = [100 * np.asarray(c, float) for c in columns]
    x = np.arange(len(vals), dtype=float)
    jitter = ((np.arange(len(vals[0])) % 7) - 3) * 0.04
    for i in range(len(vals) - 1):
        for ya, yb, jj in zip(vals[i], vals[i + 1], jitter):
            ax.plot([x[i] + jj, x[i + 1] + jj], [ya, yb], color="#c8c8c8", lw=0.5, zorder=1)
    for i, (v, c) in enumerate(zip(vals, colors)):
        bp = ax.boxplot(v, positions=[x[i]], widths=0.5, patch_artist=True, showfliers=False, zorder=2,
                        medianprops=dict(color=c, lw=1.4), whiskerprops=dict(color=c, lw=0.9),
                        capprops=dict(color=c, lw=0.9), boxprops=dict(edgecolor=c, lw=1.0))
        bp["boxes"][0].set_facecolor(c)
        bp["boxes"][0].set_alpha(0.30)
        bp["boxes"][0].set_edgecolor(c)
        ax.scatter(x[i] + jitter, v, s=9, color=c, edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(x[i], 103, f"{v.mean():.1f} ± {v.std(ddof=1):.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    ax.set_ylim(min(0.0, np.floor(min(v.min() for v in vals) / 10) * 10), 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Neuron ID accuracy (%)")


def paired_p(a, b):
    """Two-sided Wilcoxon signed-rank P of paired values (NaN if every pair is equal)."""
    from scipy.stats import wilcoxon
    return float(wilcoxon(a, b).pvalue) if (np.asarray(a) != np.asarray(b)).any() else float("nan")


def plot_compare(df, weights, path):
    plt = _style()
    p = paired_p(df.edge_trust, df.baseline)
    fig, ax = plt.subplots(figsize=(2.7, 3.4))
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.09, top=0.775)
    paired_panel(ax, [df.baseline, df.edge_trust], ["Baseline", "Edge trust"], [ORANGE, BLUE])
    w = ", ".join(f"{v:g}" for v in weights)
    _header(fig, ax, [f"Leave-one-out, n = {len(df)} worms", f"Term weights {w}",
                      f"Wilcoxon signed-rank P = {p:.2g}"])
    fig.savefig(path)
    plt.close(fig)
    return p


def plot_fig3a(df, path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.0, 3.2))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.09, top=0.845)
    paired_panel(ax, [df.openworm, df.neuropal, df.ours], ["OpenWorm", "NeuroPAL", "Data-driven"], [GRAY, RED, BLUE])
    _header(fig, ax, [f"Fig. 3a atlases, leave-one-out, n = {len(df)} worms"])
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("all", "compare", "fig3a"), default="all")
    ap.add_argument("--weights", type=float, nargs=5, default=[1.0] * 5, metavar=("AP", "LR", "DV", "DIST", "ANGLE"),
                    help="term weights for the baseline vs edge-trust comparison (default: all 1)")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                    help="parallel processes (default: up to 4)")
    args = ap.parse_args()
    parts = ("compare", "fig3a") if args.part == "all" else (args.part,)
    weights = tuple(args.weights)

    D = load()
    n = len(D["worms"])
    print(f"{n} inx-4p & mbr-1p worms, {len(D['scored'])} scored identities, {args.workers} process(es)")
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(run_fold, range(n), [parts] * n, [weights] * n))
    else:
        res = [run_fold(k, parts, weights) for k in range(n)]
    df = pd.DataFrame(res)
    print(f"done in {time.time() - t0:.0f} s\n")
    RESULTS.mkdir(exist_ok=True)

    if "compare" in parts:
        cmp_ = df[["worm", "n_scored", "baseline", "edge_trust"]]
        cmp_.to_csv(RESULTS / "baseline_vs_edge_trust.csv", index=False, lineterminator="\n")
        p = plot_compare(cmp_, weights, RESULTS / "baseline_vs_edge_trust.png")
        print(f"Baseline vs edge-trust CRF-ID, weights {weights} (mean +/- s.d. over {n} worms):")
        for col, name in (("baseline", "baseline"), ("edge_trust", "edge trust")):
            print(f"  {name:10s} {100 * cmp_[col].mean():5.1f} +/- {100 * cmp_[col].std():4.1f} %")
        d = 100 * (cmp_.edge_trust - cmp_.baseline)
        print(f"  difference {d.mean():+5.1f} pp (edge trust higher in {int((d > 0).sum())}, equal in "
              f"{int((d == 0).sum())}, lower in {int((d < 0).sum())} worms); Wilcoxon signed-rank P = {p:.2g}\n")

    if "fig3a" in parts:
        exp = read("fig3a_inx4_expected.csv").set_index("worm").reindex(df.worm)
        f3 = df[["worm", "n_scored", "ours", "neuropal", "openworm"]]
        f3.to_csv(RESULTS / "fig3a_inx4.csv", index=False, lineterminator="\n")
        plot_fig3a(f3, RESULTS / "fig3a_inx4.png")
        assert (exp.n_scored.to_numpy() == f3.n_scored.to_numpy()).all()
        print(f"Fig. 3a, inx-4p & mbr-1p (mean over {n} worms; stored value in brackets):")
        ok = True
        for col, name in (("ours", "data-driven"), ("neuropal", "NeuroPAL"), ("openworm", "OpenWorm")):
            same = np.isclose(f3[col].to_numpy(), exp[col].to_numpy(), rtol=0, atol=1e-12)
            ok &= bool(same.all())
            print(f"  {name:12s} {100 * f3[col].mean():5.1f} %  [{100 * exp[col].mean():5.1f} %]   "
                  f"per-worm values identical in {int(same.sum())}/{n} worms")
        print("  reproduction: " + ("all per-worm values match" if ok else "MISMATCH, see fig3a_inx4.csv"))
    print(f"\nresults written to {RESULTS}")


if __name__ == "__main__":
    main()
