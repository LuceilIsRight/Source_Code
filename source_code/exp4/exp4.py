"""Experiment 4: Monte Carlo integration of a normalised Gaussian in 1D, 2D, 3D."""
import json
import numpy as np
from math import erf, sqrt, pi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(4)
L = 5.0                                   # integration domain [-L, L]^d (L = 5 sigma)
Nmax = 10**6

def g(x, d):
    """Normalised Gaussian, sigma = 1: g = (2 pi)^(-d/2) exp(-|x|^2 / 2)."""
    return (2 * pi) ** (-d / 2) * np.exp(-0.5 * (x**2).sum(axis=1))

def running(d, N):
    """Running MC estimate I_N = V * mean(g) with uniform sampling in the cube."""
    x = rng.uniform(-L, L, (N, d))
    return (2 * L) ** d * np.cumsum(g(x, d)) / np.arange(1, N + 1)

exact = {d: erf(L / sqrt(2)) ** d for d in (1, 2, 3)}     # integral over the cube
pts = np.unique(np.logspace(1, 6, 26).astype(int))
runs = 30
res = {"exact": exact, "L": L, "runs": runs, "Nmax": Nmax}
fig, ax0 = plt.subplots(figsize=(6.5, 4.2)); ax = [ax0, None]
fig2, ax2 = plt.subplots(figsize=(6, 4.2))
Ns = np.arange(1, Nmax + 1)
tab = {}
for d in (1, 2, 3):
    est = running(d, Nmax)
    ax[0].semilogx(Ns[::20], est[::20], lw=.8, label=f"{d}D")
    ax[0].axhline(exact[d], color="k", ls=":", lw=.6)
    # rms error over independent runs, compared with theory sigma_I = sqrt(V (4 pi)^(-d/2) - I^2)/sqrt(N)
    e = np.array([running(d, Nmax)[pts - 1] - exact[d] for _ in range(runs)])
    rms = np.sqrt((e**2).mean(axis=0))
    var1 = (2 * L) ** d * (4 * pi) ** (-d / 2) - exact[d] ** 2     # variance of one sample
    ax2.loglog(pts, rms, "o-", ms=3, label=f"{d}D simulation")
    ax2.loglog(pts, np.sqrt(var1 / pts), "--", color=f"C{d-1}", lw=.8)
    tab[d] = dict(est=float(est[-1]), err=float(est[-1] - exact[d]),
                  rms_1e6=float(rms[-1]), theory_1e6=float(np.sqrt(var1 / 1e6)),
                  est_1e3=float(est[999]), est_1e4=float(est[9999]), est_1e5=float(est[99999]))
ax[0].axhline(1.0, color="gray", lw=.5)
ax[0].set_ylim(0.9, 1.1); ax[0].set_xlabel("number of samples N"); ax[0].set_ylabel("estimated integral")
ax[0].set_title("MC estimate of the Gaussian integral"); ax[0].legend(); ax[0].grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e4_est.png", dpi=140); plt.close(fig)
ax2.plot([], [], "k--", lw=.8, label=r"theory $\sigma_1/\sqrt{N}$")
ax2.set_xlabel("number of samples N"); ax2.set_ylabel("rms error of integral")
ax2.set_title("Error scaling (30 independent runs)"); ax2.legend(fontsize=8); ax2.grid(alpha=.3, which="both")
fig2.tight_layout(); fig2.savefig("figs/e4_err.png", dpi=140); plt.close(fig2)
res["tab"] = tab
json.dump(res, open("res/e4.json", "w"), indent=1)
print(res)
