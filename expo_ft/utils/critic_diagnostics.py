"""Offline critic diagnosis on the online replay buffer (train_pi_robo.py --diagnose_critic_out=DIR).

For every episode in the buffer, at each planning point t (every replan_steps env steps):
  q_exec   Q(s_t, executed chunk) for each ensemble member (mean / std over members)
  q_cand   Q(s_t, c) for N fresh pi0.5 candidate chunks c (what rollout selection compares)
  G_t      the discounted return that actually followed (gamma per env step; 1 at a success)

Writes DIR/report.txt + DIR/diagnosis.npz and prints the report; plots (Q along every episode,
summary) come from scripts/dk1_task/plot_critic_diagnosis.py DIR. The report:
  - Spearman(Q_exec, G): does Q track the real outcome?            (> ~0.3 = learning something)
  - success vs failure Q at the same point of the episode
  - candidate spread / ensemble std: is "best of N" above critic noise? (< 1 = ~random pick)
  - executed-chunk percentile among candidates, in successes vs failures
Read-only: no checkpoint, buffer or wandb writes.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from expo_ft.agents.alg.batch_utils import prepare_actor_sampling_batch, prepare_critic_batch


def _episodes(rb):
    n = len(rb)
    dones = np.asarray(rb.dataset_dict["dones"][:n]).astype(bool)
    rewards = np.asarray(rb.dataset_dict["rewards"][:n])
    eps, start = [], 0
    for e in np.flatnonzero(dones):
        eps.append((start, int(e), bool(rewards[e] > 0.5)))
        start = int(e) + 1
    return eps


def _spearman(a, b):
    ra, rb_ = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb_)[0, 1]) if len(a) > 2 else float("nan")


def _q_all(agent, enc, actions, states):
    """(num_qs, B) Q-values of every ensemble member."""
    return agent.critic.apply_fn({"params": agent.critic.params}, enc, actions, False, p=states)


def run(agent, rb, out_dir, chunk=8, gamma=0.99, max_episodes=None, seed=0):
    from expo_ft.agents.alg.expo_ft import _images_to_float, batch_encode

    os.makedirs(out_dir, exist_ok=True)
    R, N = agent.replan_steps, agent.N
    eps = _episodes(rb)
    if max_episodes:
        eps = eps[-max_episodes:]
    max_start = len(rb) - R
    print(f"[diag] {len(eps)} episodes ({sum(e[2] for e in eps)} successes), N={N}, replan={R}", flush=True)

    rng = jax.random.PRNGKey(seed)
    per_ep = []
    for k, (s, e, succ) in enumerate(eps):
        ts = np.arange(s, e + 1, R)
        ts = ts[ts < max_start]
        if len(ts) == 0:
            continue
        g = np.where(succ, gamma ** (e - ts).astype(np.float64), 0.0)
        qe_m, qe_s, qc = [], [], []
        for i in range(0, len(ts), chunk):
            idx = ts[i:i + chunk]
            b = rb._convert_to_openpi_format(rb.sample_jax(len(idx), indices=idx))
            b = _images_to_float(b)
            b = prepare_critic_batch(b, agent.actor.model_config.action_dim, agent.action_dim, agent.state_dim,
                                     agent.action_horizon, R, agent.critic_camera_keys)
            enc = batch_encode(agent.batch_encoder.apply_fn, agent.batch_encoder.params, b["observations"],
                               stop_gradient=True)
            q = np.asarray(_q_all(agent, enc, b["actions"], b["critic_states"]))  # (Q, B)
            qe_m.append(q.mean(0))
            qe_s.append(q.std(0))
            # N fresh candidates at s_t (the rollout comparison): actor sampling on the current obs
            cur = dict(b, next_image=b["image"], next_image_mask=b["image_mask"], next_states=b["states"])
            rng, key = jax.random.split(rng)
            acts, _ = agent.actor.sample_training_actions(
                transformed_inputs=prepare_actor_sampling_batch(cur), train_state=agent.actor_train_state,
                rng=key, train=False, num_samples=N)
            acts = acts[:, :R, :].reshape(len(idx) * N, -1)
            qcand = np.asarray(_q_all(agent, jnp.repeat(enc, N, axis=0), acts,
                                      jnp.repeat(b["critic_states"], N, axis=0)))  # (Q, B*N)
            qc.append(qcand.reshape(qcand.shape[0], len(idx), N).transpose(1, 0, 2))  # (B, Q, N)
        per_ep.append(dict(start=s, end=e, success=succ, t=ts - s, G=g, q_m=np.concatenate(qe_m),
                           q_s=np.concatenate(qe_s), q_cand=np.concatenate(qc)))
        print(f"[diag] episode {k + 1}/{len(eps)} ({'S' if succ else 'F'}, {e - s + 1} steps)", flush=True)

    _report(per_ep, out_dir)
    return per_ep


def _report(per_ep, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    q = np.concatenate([p["q_m"] for p in per_ep])
    G = np.concatenate([p["G"] for p in per_ep])
    cand_m = np.concatenate([p["q_cand"].mean(1) for p in per_ep])           # (P, N) ensemble-mean per candidate
    cand_ens = np.concatenate([p["q_cand"].std(1).mean(-1) for p in per_ep])  # (P,) member disagreement
    spread = cand_m.max(1) - cand_m.min(1)
    cand_sd = cand_m.std(1)
    pct = np.concatenate([(p["q_cand"].mean(1) < p["q_m"][:, None]).mean(1) for p in per_ep])
    succ_pt = np.concatenate([np.full(len(p["t"]), p["success"]) for p in per_ep])
    rel = np.concatenate([p["t"] / max(p["end"] - p["start"], 1) for p in per_ep])

    lines = [
        f"planning points: {len(q)} in {len(per_ep)} episodes ({sum(p['success'] for p in per_ep)} successes)",
        f"Q_exec: mean {q.mean():.3f}, range [{q.min():.3f}, {q.max():.3f}]",
        f"Spearman(Q_exec, actual discounted return G): {_spearman(q, G):.3f}",
    ]
    for lo, hi in ((0, .33), (.33, .66), (.66, 1.01)):
        m = (rel >= lo) & (rel < hi)
        qs, qf = q[m & succ_pt], q[m & ~succ_pt]
        lines.append(f"  episode {int(lo * 100):3d}-{int(min(hi, 1) * 100):3d}%: Q success {qs.mean() if len(qs) else np.nan:.3f}"
                     f" vs failure {qf.mean() if len(qf) else np.nan:.3f}")
    lines += [
        f"candidates: spread (max-min) {np.median(spread):.4f}, std {np.median(cand_sd):.4f} | "
        f"ensemble std {np.median(cand_ens):.4f} -> std/ensemble = {np.median(cand_sd / np.maximum(cand_ens, 1e-9)):.2f} "
        f"(< 1: the best-of-N pick is mostly critic noise)",
        f"executed chunk above this fraction of fresh candidates: successes {pct[succ_pt].mean():.2f}, "
        f"failures {pct[~succ_pt].mean():.2f} (0.5 = no preference)",
    ]
    report = "\n".join(lines)
    print("\n[diag] " + report.replace("\n", "\n[diag] "), flush=True)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(report + "\n")
    np.savez(os.path.join(out_dir, "diagnosis.npz"),
             ep=np.concatenate([np.full(len(p["t"]), k) for k, p in enumerate(per_ep)]),
             t=np.concatenate([p["t"] for p in per_ep]), G=G, q=q,
             q_std=np.concatenate([p["q_s"] for p in per_ep]),
             q_cand=np.concatenate([p["q_cand"] for p in per_ep]),  # (P, members, N)
             success=succ_pt, rel=rel,
             ep_bounds=np.array([[p["start"], p["end"], p["success"]] for p in per_ep]))
    print(f"[diag] wrote {out_dir}/report.txt and diagnosis.npz; plots: "
          f"uv run --with matplotlib --with numpy python scripts/dk1_task/plot_critic_diagnosis.py {out_dir}",
          flush=True)
