"""Experiment 3: Area of a circle by Monte Carlo (hit-or-miss) sampling."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
R = 1.0                                  # circle radius
A_square = (2 * R) ** 2                  # bounding square [-R, R]^2
A_exact = np.pi * R**2
Nmax = 10**6

def running_estimate(N):
    """Area estimate after every sample 1..N for one random sequence."""
    x = rng.uniform(-R, R, N); y = rng.uniform(-R, R, N)
    inside = (x**2 + y**2) <= R**2
    return A_square * np.cumsum(inside) / np.arange(1, N + 1), x, y, inside

est, x, y, inside = running_estimate(Nmax)

# scatter illustration with 2000 points
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
k = 2000
ax[0].scatter(x[:k][inside[:k]], y[:k][inside[:k]], s=2, c="tab:blue", label="inside")
ax[0].scatter(x[:k][~inside[:k]], y[:k][~inside[:k]], s=2, c="tab:red", label="outside")
th = np.linspace(0, 2 * np.pi, 300); ax[0].plot(np.cos(th), np.sin(th), "k", lw=.8)
ax[0].set_aspect("equal"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[0].set_title("Hit-or-miss sampling (2000 points)"); ax[0].legend(markerscale=4, fontsize=8, loc="upper right")
Ns = np.arange(1, Nmax + 1)
ax[1].semilogx(Ns, est, lw=.8, label="Monte Carlo estimate")
ax[1].axhline(A_exact, color="k", ls="--", label=r"exact $\pi R^2$")
ax[1].set_ylim(2.6, 3.7); ax[1].set_xlabel("number of samples N"); ax[1].set_ylabel("estimated area")
ax[1].set_title("Convergence of the estimate"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e3_area.png", dpi=140); plt.close(fig)

# error versus N: single run and rms over 40 independent runs
pts = np.unique(np.logspace(1, 6, 26).astype(int))
runs = 40
err = np.empty((runs, pts.size))
for r in range(runs):
    e, *_ = running_estimate(Nmax)
    err[r] = e[pts - 1] - A_exact
rms = np.sqrt((err**2).mean(axis=0))
p = np.pi / 4
sigma = A_square * np.sqrt(p * (1 - p)) / np.sqrt(pts)        # analytic standard error
slope = np.polyfit(np.log(pts), np.log(rms), 1)[0]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(Ns[::50], np.abs(est[::50] - A_exact), ".", ms=2, alpha=.5, label="|error|, single run")
ax.loglog(pts, rms, "o-", label=f"rms error, 40 runs (slope {slope:.2f})")
ax.loglog(pts, sigma, "k--", label=r"theory $4\sqrt{p(1-p)}/\sqrt{N}$")
ax.set_xlabel("number of samples N"); ax.set_ylabel("absolute error in area")
ax.set_title("Error scaling"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig("figs/e3_error.png", dpi=140); plt.close(fig)

table = [(int(n_), float(est[n_ - 1]), float(est[n_ - 1] - A_exact)) for n_ in (10, 100, 1000, 10**4, 10**5, 10**6)]
res = dict(A_exact=A_exact, table=table, slope=float(slope), runs=runs, Nmax=Nmax,
           coeff=float(A_square * np.sqrt(p * (1 - p))))
json.dump(res, open("res/e3.json", "w"), indent=1)
print(res)
