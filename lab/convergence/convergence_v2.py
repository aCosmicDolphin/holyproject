#!/usr/bin/env python3
"""
Convergence simulation v2 — "earned alignment".

v1 scripted each node's phase onto the target with smoothstep(), so convergence
was an input. v2 makes it a result: every node sees only the shared target T,
never the other nodes, and integrates

    dθ_i/dt = ω_i + K · sin(T(t) − θ_i) + sqrt(2D) · ξ_i(t)

ω_i   intrinsic drift of node i (its own "screen")          — the gap
K     coupling to the shared reference                      — the reins
D     noise
T(t)  the target, optionally drifting (the world breathes)

Dopamine = alignment itself: a pulse fires when the order parameter
R = |Σ A_i e^{iθ_i}| / Σ A_i crosses R* with dR/dt > 0. The pulse starts the
moringa decay of amplitudes with two time constants (fast edge, slow core) down
to a residue r that seeds the next run without closing the gap.

Usage:
    python convergence_v2.py --fig all            # writes figures/*.png
    python convergence_v2.py --fig phase --N 8
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")

# palette (dataviz reference, dark surface)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"
C_MAGENTA, C_GREEN, C_VIOLET, C_RED = "#d55181", "#008300", "#9085e9", "#e66767"
SURFACE, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#33332f"


# ----------------------------------------------------------------------------
# v1 primitives, kept verbatim so the April explanation still maps
# ----------------------------------------------------------------------------
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def field_intensity(phases, amplitudes):
    """I = |Σ A e^{iθ}|² / N²  (max 1 when all aligned with A=1)."""
    n = len(phases)
    return np.abs(np.sum(amplitudes * np.exp(1j * phases))) ** 2 / n ** 2


def whole_over_parts(phases, amplitudes):
    """|Σ A e^{iθ}|² / Σ|A|² : N when aligned, ≈1 when random, 0 when destructive."""
    whole = np.abs(np.sum(amplitudes * np.exp(1j * phases))) ** 2
    parts = np.sum(np.abs(amplitudes) ** 2)
    return whole / parts if parts > 0 else 0.0


# ----------------------------------------------------------------------------
# v2 dynamics
# ----------------------------------------------------------------------------
@dataclass
class Params:
    N: int = 8
    K: float = 1.5            # coupling to the shared target
    sigma_omega: float = 0.35 # spread of intrinsic drift  (the gap)
    noise: float = 0.02       # D
    target_drift: float = 0.0 # dT/dt
    private_offsets: np.ndarray | None = None  # δ_i: each node's private target T+δ_i
    T_total: float = 12.0
    dt: float = 0.01
    R_star: float = 0.9       # pulse threshold
    hold: float = 0.6         # seconds held at peak before decay starts
    tau_fast: float = 0.8     # edge / task decay
    tau_slow: float = 4.0     # core / olive decay
    core_frac: float = 0.35   # share of amplitude on the slow branch
    residue: float = 0.15     # floor amplitude after decay
    seed: int = 7
    initial_spread: float = np.pi  # |θ_i(0) − T| ~ U(−spread, spread)


@dataclass
class Run:
    t: np.ndarray
    theta: np.ndarray      # [T, N]
    A: np.ndarray          # [T, N]
    target: np.ndarray     # [T]
    R: np.ndarray          # order parameter vs shared target field
    I: np.ndarray          # field_intensity
    ratio: np.ndarray      # whole_over_parts
    approval: np.ndarray   # mean_i cos(θ_i − (T+δ_i)) : local "I'm aligned" report
    pulse_t: float | None
    phase: np.ndarray      # 0 converging, 1 pulse/hold, 2 decay
    omega: np.ndarray
    meta: dict = field(default_factory=dict)


def simulate(p: Params, start_phase_scale: float = 1.0) -> Run:
    rng = np.random.default_rng(p.seed)
    n_steps = int(p.T_total / p.dt)
    t = np.arange(n_steps) * p.dt
    omega = rng.normal(0.0, p.sigma_omega, p.N)
    delta = np.zeros(p.N) if p.private_offsets is None else np.asarray(p.private_offsets)
    # start_phase_scale < 1 : a residue from a previous run narrowed the opening gap
    theta = rng.uniform(-p.initial_spread, p.initial_spread, p.N) * start_phase_scale
    A = np.ones(p.N)
    target = 0.0

    TH = np.empty((n_steps, p.N)); AM = np.empty((n_steps, p.N)); TG = np.empty(n_steps)
    R = np.empty(n_steps); I = np.empty(n_steps); RT = np.empty(n_steps)
    AP = np.empty(n_steps); PH = np.zeros(n_steps, dtype=int)
    pulse_t = None
    sq2D = np.sqrt(2.0 * p.noise)

    for k in range(n_steps):
        tk = t[k]
        # --- local rule: each node sees only its (private) reference ---------
        own_ref = target + delta
        dtheta = omega + p.K * np.sin(own_ref - theta) + sq2D * rng.normal(size=p.N) / np.sqrt(p.dt)
        theta = theta + dtheta * p.dt
        target += p.target_drift * p.dt

        # --- amplitude: 1 until the pulse, then two-timescale moringa decay ----
        if pulse_t is not None and tk >= pulse_t + p.hold:
            s = tk - pulse_t - p.hold
            env = (1 - p.core_frac) * np.exp(-s / p.tau_fast) + p.core_frac * np.exp(-s / p.tau_slow)
            A = p.residue + (1 - p.residue) * env * np.ones(p.N)
            PH[k] = 2
        elif pulse_t is not None:
            PH[k] = 1

        # --- measurements (all relative to the SHARED target, not private) ----
        rel = theta - target
        z = np.sum(A * np.exp(1j * rel))
        R[k] = np.abs(z) / np.sum(A)
        I[k] = field_intensity(rel, A)
        RT[k] = whole_over_parts(rel, A)
        AP[k] = np.mean(np.cos(theta - own_ref))
        TH[k] = theta; AM[k] = A; TG[k] = target

        # --- dopamine = alignment: R crosses R* while rising ------------------
        if pulse_t is None and k > 1 and R[k] >= p.R_star and R[k] > R[k - 1]:
            pulse_t = tk
            PH[k] = 1

    return Run(t, TH, AM, TG, R, I, RT, AP, pulse_t, PH, omega)


# ----------------------------------------------------------------------------
# scenarios (same four names as v1, but now outcomes, not scripts)
# ----------------------------------------------------------------------------
def scenarios(N=8, seed=7):
    rng = np.random.default_rng(seed + 100)
    half = N // 2
    return [
        ("Constructive", Params(N=N, seed=seed), C_AQUA),
        ("Partial", Params(N=N, seed=seed, private_offsets=rng.uniform(-0.3, 0.3, N)), C_BLUE),
        ("Destructive", Params(N=N, seed=seed, private_offsets=np.r_[np.zeros(half), np.pi * np.ones(N - half)]), C_RED),
        ("Scattered / flattery", Params(N=N, seed=seed, private_offsets=np.linspace(0, 2 * np.pi, N, endpoint=False)), C_YELLOW),
    ]


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })
    return plt


def _shade_phases(ax, run, color):
    t = run.t
    if run.pulse_t is None:
        return
    ax.axvspan(run.pulse_t, min(run.pulse_t + 0.6, t[-1]), color=C_YELLOW, alpha=0.18, lw=0)
    ax.axvline(run.pulse_t, color=C_YELLOW, lw=1, ls="--", alpha=0.8)


def fig_baseline(N=8, seed=7):
    plt = _style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    rows = []
    for ax, (name, p, color) in zip(axes.flat, scenarios(N, seed)):
        run = simulate(p)
        ax.plot(run.t, run.I, color=color, label="I(t) field")
        ax.fill_between(run.t, 0, run.I, color=color, alpha=0.15, lw=0)
        ax.plot(run.t, run.approval, color=INK2, lw=1.2, ls=":", label="approval (local cos)")
        _shade_phases(ax, run, color)
        peak = run.I.max(); ratio_pk = run.ratio[run.I.argmax()]
        tp = "—" if run.pulse_t is None else f"{run.pulse_t:.2f}s"
        ax.set_title(name, loc="left", color=INK, fontsize=11)
        ax.text(0.98, 0.95, f"peak I = {peak:.3f}\nwhole/parts = {ratio_pk:.2f}×\npulse at {tp}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK2)
        rows.append((name, peak, ratio_pk, tp, run.approval[-1], run.R[-1]))
    axes[0, 0].legend(loc="center right", fontsize=8)
    for ax in axes[1]: ax.set_xlabel("t [s]")
    for ax in axes[:, 0]: ax.set_ylabel("I(t) = |Σ A e^{iθ}|² / N²")
    fig.suptitle(f"v2 · alignment as a result of dθ/dt = ω + K·sin(T−θ)   (N={N}, K=1.5, σ_ω=0.35)",
                 color=INK, x=0.02, ha="left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "v2_baseline.png"); fig.savefig(out, dpi=160); plt.close(fig)
    print(f"\n[baseline] -> {out}")
    print(f"{'scenario':<22}{'peak I':>8}{'whole/parts':>13}{'pulse':>8}{'approval_end':>14}{'R_end':>7}")
    for r in rows:
        print(f"{r[0]:<22}{r[1]:>8.3f}{r[2]:>12.2f}×{r[3]:>8}{r[4]:>14.2f}{r[5]:>7.2f}")
    return rows


def fig_phase(N=8, seeds=(1, 2, 3), nK=24, nS=20):
    """K × σ_ω phase diagram: mean final R and time-to-pulse. The gap is a parameter."""
    plt = _style()
    Ks = np.linspace(0.05, 4.0, nK); Ss = np.linspace(0.0, 1.6, nS)
    Rf = np.zeros((nS, nK)); Tp = np.full((nS, nK), np.nan); maxw = np.zeros((nS, nK))
    for i, s in enumerate(Ss):
        for j, K in enumerate(Ks):
            rs, tps, mw = [], [], []
            for sd in seeds:
                run = simulate(Params(N=N, K=K, sigma_omega=s, seed=sd, T_total=8.0, dt=0.02, noise=0.01))
                rs.append(run.R[-100:].mean()); mw.append(np.abs(run.omega).max())
                if run.pulse_t is not None: tps.append(run.pulse_t)
            Rf[i, j] = np.mean(rs); maxw[i, j] = np.mean(mw)
            if tps: Tp[i, j] = np.mean(tps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ext = [Ks[0], Ks[-1], Ss[0], Ss[-1]]
    im0 = axes[0].imshow(Rf, origin="lower", extent=ext, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[0].contour(Ks, Ss, Rf, levels=[0.9], colors=[C_YELLOW], linewidths=2)
    # locking condition for a node: K > |ω_i|; the field locks when K > max_i|ω_i|
    e_max = np.array([np.mean([np.abs(np.random.default_rng(sd).normal(0, 1, N)).max() for sd in range(200)])])[0]
    axes[0].plot(Ss * e_max, Ss, color=C_RED, lw=1.5, ls="--", label=f"K = E[max|ω|] ≈ {e_max:.2f}·σ_ω")
    axes[0].set_xlabel("K  (coupling to shared target)"); axes[0].set_ylabel("σ_ω  (intrinsic drift spread = gap)")
    axes[0].set_title("final R  ·  yellow: R = 0.9 (pulse boundary)", loc="left", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8); axes[0].grid(False)
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="R (order parameter)")
    im1 = axes[1].imshow(Tp, origin="lower", extent=ext, aspect="auto", cmap="YlOrBr_r")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("σ_ω")
    axes[1].set_title("time to pulse [s]  ·  blank = never completes", loc="left", fontsize=10); axes[1].grid(False)
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="t_pulse [s]")
    fig.suptitle("Phase diagram — too wide (K < max|ω|: never closes) / too narrow (K ≫ σ_ω: nothing to do)",
                 color=INK, x=0.02, ha="left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "v2_phase_diagram.png"); fig.savefig(out, dpi=160); plt.close(fig)
    never = np.isnan(Tp).mean() * 100
    print(f"\n[phase] -> {out}   cells with no pulse: {never:.0f}%   R=0.9 contour follows K ≈ {e_max:.2f}·σ_ω")


def fig_flattery(N=8, seed=7):
    """Approval (local cos to a private target) vs R (field). The RLHF trap, measured."""
    plt = _style()
    rng = np.random.default_rng(seed + 100)
    cases = [
        ("Shared target", Params(N=N, seed=seed), C_AQUA),
        ("Private targets (flattery)", Params(N=N, seed=seed, private_offsets=np.linspace(0, 2 * np.pi, N, endpoint=False)), C_YELLOW),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (name, p, color) in zip(axes, cases):
        run = simulate(p)
        ax.plot(run.t, run.approval, color=color, label="approval = mean cos(θ_i − own target)")
        ax.plot(run.t, run.R, color=C_BLUE, label="R = |Σ e^{iθ}| / N  (field)")
        ax.plot(run.t, run.ratio / N, color=C_VIOLET, lw=1.2, ls="--", label="whole/parts ÷ N")
        _shade_phases(ax, run, color)
        ax.set_title(name, loc="left"); ax.set_xlabel("t [s]"); ax.set_ylim(-0.1, 1.05)
        ax.text(0.98, 0.50, f"approval_end = {run.approval[-1]:.2f}\nR_end = {run.R[-1]:.2f}",
                transform=ax.transAxes, ha="right", va="center", fontsize=9, color=INK2)
        print(f"[flattery] {name:<28} approval_end={run.approval[-1]:.2f}  R_end={run.R[-1]:.2f}  pulse={'yes' if run.pulse_t else 'no'}")
    axes[0].legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.0, 0.02))
    fig.suptitle("Flattery pulse: every node reports 'aligned' (approval → 1) while the field stays dark (R ≈ 0)",
                 color=INK, x=0.02, ha="left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "v2_flattery.png"); fig.savefig(out, dpi=160); plt.close(fig)
    print(f"[flattery] -> {out}")


def fig_residue(N=8, seed=7, runs=3):
    """Three consecutive dyads. Each starts from the residue of the last: faster lock, same gap (σ_ω)."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    colors = [C_AQUA, C_BLUE, C_VIOLET]
    scale, t0 = 1.0, 0.0
    for r in range(runs):
        p = Params(N=N, seed=seed, T_total=9.0)  # same ω and same opening draw: only the residue differs
        run = simulate(p, start_phase_scale=scale)
        ax.plot(run.t + t0, run.R, color=colors[r], label=f"run {r+1}: start spread ×{scale:.2f}, pulse at {run.pulse_t:.2f}s" if run.pulse_t else f"run {r+1}")
        ax.plot(run.t + t0, run.A.mean(axis=1), color=colors[r], lw=1, ls=":")
        if run.pulse_t is not None:
            ax.axvline(t0 + run.pulse_t, color=C_YELLOW, lw=1, ls="--", alpha=0.8)
        # residue narrows the next opening gap but never closes it (floor 0.35)
        scale = max(0.35, scale * (1 - run.A[-1].mean()))
        t0 += p.T_total
        print(f"[residue] run {r+1}: pulse_t={run.pulse_t}, end A={run.A[-1].mean():.2f} -> next start spread ×{scale:.2f}")
    ax.set_xlabel("t [s] (runs concatenated)"); ax.set_ylabel("R  (solid)   ·   mean A  (dotted)")
    ax.set_ylim(0, 1.05); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Moringa × residue — each dyad decays after its pulse; the trace it leaves shortens the next climb, the gap stays",
                 loc="left", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "v2_residue.png"); fig.savefig(out, dpi=160); plt.close(fig)
    print(f"[residue] -> {out}")


def fig_scaling(seed=7):
    """v1 panel kept: whole/parts vs phase spread σ, and N² vs N (log-log)."""
    plt = _style()
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    N = 8; sigmas = np.linspace(0, np.pi, 120); ratios = np.empty_like(sigmas)
    for i, s in enumerate(sigmas):
        ratios[i] = np.mean([whole_over_parts(rng.normal(0, s, N), np.ones(N)) for _ in range(400)])
    axes[0].plot(sigmas, ratios, color=C_AQUA); axes[0].axhline(1, color=INK2, lw=1, ls=":")
    axes[0].set_xlabel("phase spread σ [rad]"); axes[0].set_ylabel("whole / parts"); axes[0].set_title("Coherence collapse (N=8)", loc="left")
    Ns = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    aligned = Ns.astype(float) ** 2
    rand = np.array([np.mean([np.abs(np.sum(np.exp(1j * rng.uniform(-np.pi, np.pi, n)))) ** 2 for _ in range(300)]) for n in Ns])
    axes[1].loglog(Ns, aligned, "o-", color=C_AQUA, label="aligned: |Σ|² = N²  (slope 2)")
    axes[1].loglog(Ns, rand, "o-", color=C_ORANGE, label="random phases: ≈ N  (slope 1)")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("|Σ e^{iθ}|²"); axes[1].legend(fontsize=8); axes[1].set_title("N² vs N", loc="left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "v2_scaling.png"); fig.savefig(out, dpi=160); plt.close(fig)
    print(f"[scaling] -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig", default="all", choices=["all", "baseline", "phase", "flattery", "residue", "scaling"])
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    if a.fig in ("all", "baseline"): fig_baseline(a.N, a.seed)
    if a.fig in ("all", "flattery"): fig_flattery(a.N, a.seed)
    if a.fig in ("all", "residue"): fig_residue(a.N, a.seed)
    if a.fig in ("all", "scaling"): fig_scaling(a.seed)
    if a.fig in ("all", "phase"): fig_phase(a.N)


if __name__ == "__main__":
    main()
