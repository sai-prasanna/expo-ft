"""Plots for a critic diagnosis (train_pi_robo.py --diagnose_critic_out=DIR). No project imports.

    uv run --with matplotlib --with numpy python scripts/dk1_task/plot_critic_diagnosis.py DIR

DIR/episodes.png: Q(executed chunk) along every episode (green success / red failure, band = ensemble
std), the min-max of N fresh pi0.5 candidates (grey; the spread the rollout pick chooses within) and
the actual discounted return (dashed, right axis).
DIR/summary.png: Q vs actual return; success vs failure along the episode; candidate std vs
ensemble std.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def main(out_dir):
    d = np.load(os.path.join(out_dir, "diagnosis.npz"))
    ep, t, G, q, q_std, q_cand, succ, rel = (d[k] for k in ("ep", "t", "G", "q", "q_std", "q_cand", "success", "rel"))
    cand_m = q_cand.mean(1)            # (P, N): ensemble-mean Q per candidate
    cand_sd = cand_m.std(1)
    cand_ens = q_cand.std(1).mean(-1)  # member disagreement
    n_ep = int(ep.max()) + 1
    lo_y, hi_y = min(q.min(), cand_m.min()), max(q.max(), cand_m.max())

    cols = 6
    rows = int(np.ceil(n_ep / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.3 * rows), sharey=True, squeeze=False)
    for k, ax in zip(range(n_ep), axes.flat):
        m = ep == k
        s = bool(succ[m][0])
        col = "tab:green" if s else "tab:red"
        ax.fill_between(t[m], cand_m[m].min(1), cand_m[m].max(1), color="0.8", label="N candidates")
        ax.plot(t[m], q[m], color=col, lw=1.5, label="Q(executed)")
        ax.fill_between(t[m], q[m] - q_std[m], q[m] + q_std[m], color=col, alpha=0.2)
        ax.set_ylim(lo_y - 0.05, hi_y + 0.05)
        ax.set_title(f"ep {k + 1} {'success' if s else 'failure'}", fontsize=8, color=col)
        ax.tick_params(labelsize=7)
        ax2 = ax.twinx()
        ax2.plot(t[m], G[m], "k--", lw=0.8)
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(labelsize=6)
    for ax in axes.flat[n_ep:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Critic Q along episodes (left axis, x = env step; band = ensemble std; grey = N fresh "
                 "pi0.5 candidates) and actual discounted return (dashed, right axis)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "episodes.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].scatter(G, q, s=6, c=np.where(succ, "tab:green", "tab:red"), alpha=0.5)
    ax[0].set_xlabel("actual discounted return G")
    ax[0].set_ylabel("Q(executed)")
    ax[0].set_title(f"Spearman {spearman(q, G):.2f}")
    bins = np.linspace(0, 1, 11)
    for flag, col, lab in ((True, "tab:green", "success"), (False, "tab:red", "failure")):
        m = succ == flag
        if m.any():
            idx = np.clip(np.digitize(rel[m], bins) - 1, 0, 9)
            ax[1].plot(bins[:-1] + 0.05, [q[m][idx == i].mean() if (idx == i).any() else np.nan for i in range(10)],
                       "o-", color=col, label=lab)
    ax[1].set_xlabel("fraction of episode")
    ax[1].set_ylabel("mean Q(executed)")
    ax[1].legend()
    ax[1].set_title("success vs failure along the episode")
    ax[2].hist(cand_sd, bins=40, alpha=0.6, label="std over N candidates")
    ax[2].hist(cand_ens, bins=40, alpha=0.6, label="ensemble std")
    ax[2].set_xlabel("Q units")
    ax[2].legend()
    ax[2].set_title("does the best-of-N choice beat critic noise?")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=110)
    plt.close(fig)
    print(f"wrote {out_dir}/episodes.png and summary.png")


if __name__ == "__main__":
    main(sys.argv[1])
