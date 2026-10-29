"""Experiment 1: Random walks in one, two and three dimensions."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)   # fixed seed -> reproducible results
N = 1000        # time steps per walk
M = 2000        # number of independent walks used for all averages
n = np.arange(1, N + 1)

def make_steps(d, M, N):
    """Return an (M, N, d) array of unit-length random steps."""
    if d == 1:                                    # +1 or -1, probability 1/2 each
        return rng.choice([-1.0, 1.0], size=(M, N, 1))
    if d == 2:                                    # uniform random angle
        th = rng.uniform(0, 2 * np.pi, (M, N))
        return np.stack([np.cos(th), np.sin(th)], axis=-1)
    z = rng.uniform(-1, 1, (M, N))                # cos(theta) uniform => isotropic
    ph = rng.uniform(0, 2 * np.pi, (M, N))
    s = np.sqrt(1 - z**2)
    return np.stack([s * np.cos(ph), s * np.sin(ph), z], axis=-1)

pos, disp, res = {}, {}, {}
for d in (1, 2, 3):
    pos[d] = np.cumsum(make_steps(d, M, N), axis=1)      # positions r(n), (M,N,d)
    # displacement: signed x(n) in 1D, distance |r(n)| from origin in 2D/3D
    disp[d] = pos[d][..., 0] if d == 1 else np.linalg.norm(pos[d], axis=-1)

# ---- (b) displacement of single walks versus N ----
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for k, d in enumerate((1, 2, 3)):
    for w in range(3):
        ax[k].plot(n, disp[d][w], lw=0.9, label=f"walk {w+1}")
    ax[k].set_title(f"{d}D walk"); ax[k].set_xlabel("number of steps N")
    ax[k].set_ylabel("displacement x(N)" if d == 1 else "displacement |r(N)|")
    ax[k].legend(fontsize=7); ax[k].grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e1_b_disp.png", dpi=140); plt.close(fig)

# trajectories (illustration)
fig = plt.figure(figsize=(8, 3.6))
a = fig.add_subplot(121)
a.plot(*pos[2][0].T, lw=.6); a.plot(0, 0, "ro", label="start")
a.set_xlabel("x"); a.set_ylabel("y"); a.set_title("2D trajectory, N=1000"); a.legend(); a.set_aspect("equal")
a = fig.add_subplot(122, projection="3d")
a.plot(*pos[3][0].T, lw=.6); a.plot([0], [0], [0], "ro", label="start")
a.set_xlabel("x"); a.set_ylabel("y"); a.set_zlabel("z"); a.set_title("3D trajectory, N=1000"); a.legend()
fig.tight_layout(); fig.savefig("figs/e1_traj.png", dpi=140); plt.close(fig)

# ---- (c) mean displacement ----
th_mean = {1: np.sqrt(2 * n / np.pi), 2: np.sqrt(np.pi * n) / 2, 3: np.sqrt(8 * n / (3 * np.pi))}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for d in (1, 2, 3):
    meanvec = np.linalg.norm(pos[d].mean(axis=0), axis=-1)      # |<r(N)>|
    meandist = disp[d].mean(axis=0) if d > 1 else np.abs(pos[d][..., 0]).mean(axis=0)
    ax[0].plot(n, meanvec, label=f"{d}D")
    ax[1].plot(n, meandist, label=f"{d}D (simulation)")
    ax[1].plot(n, th_mean[d], "k--", lw=.8, label="theory" if d == 1 else None)
    res[f"meanvec_final_{d}D"] = float(meanvec[-1])
    res[f"meandist_final_{d}D"] = float(meandist[-1])
    res[f"meandist_theory_{d}D"] = float(th_mean[d][-1])
ax[0].set_xlabel("number of steps N"); ax[0].set_ylabel(r"$|\langle \mathbf{r}(N)\rangle|$ (step lengths)")
ax[0].set_title("Mean displacement vector (averaged over walks)")
ax[1].set_xlabel("number of steps N"); ax[1].set_ylabel(r"$\langle |\mathbf{r}(N)|\rangle$ (step lengths)")
ax[1].set_title("Mean distance from origin")
for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e1_c_mean.png", dpi=140); plt.close(fig)

# ---- (d) mean square displacement ----
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
for d in (1, 2, 3):
    msd = (pos[d]**2).sum(-1).mean(axis=0)               # <|r(N)|^2>
    slope = float((msd * n).sum() / (n * n).sum())       # least squares through origin
    res[f"msd_slope_{d}D"] = slope
    res[f"msd_final_{d}D"] = float(msd[-1])
    ax.plot(n, msd, label=f"{d}D, slope={slope:.3f}")
ax.plot(n, n, "k--", lw=.8, label=r"theory $\langle r^2\rangle=N$")
ax.set_xlabel("number of steps N"); ax.set_ylabel(r"$\langle r^2(N)\rangle$ (step length$^2$)")
ax.set_title("Mean square displacement"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e1_d_msd.png", dpi=140); plt.close(fig)

# ---- (e) histograms of final displacement ----
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
x = np.linspace(-120, 120, 400)
ax[0].hist(disp[1][:, -1], bins=40, density=True, alpha=.6, label="simulation")
ax[0].plot(x, np.exp(-x**2 / (2 * N)) / np.sqrt(2 * np.pi * N), "k", label="Gaussian, $\\sigma^2=N$")
ax[0].set_xlabel("x(N=1000)"); ax[0].set_ylabel("probability density"); ax[0].set_title("1D")
r = np.linspace(0, 100, 400)
ax[1].hist(disp[2][:, -1], bins=40, density=True, alpha=.6, label="simulation")
ax[1].plot(r, 2 * r / N * np.exp(-r**2 / N), "k", label="Rayleigh")
ax[1].set_xlabel("|r(N=1000)|"); ax[1].set_ylabel("probability density"); ax[1].set_title("2D")
s2 = N / 3
ax[2].hist(disp[3][:, -1], bins=40, density=True, alpha=.6, label="simulation")
ax[2].plot(r, np.sqrt(2 / np.pi) * r**2 / s2**1.5 * np.exp(-r**2 / (2 * s2)), "k", label="Maxwell")
ax[2].set_xlabel("|r(N=1000)|"); ax[2].set_ylabel("probability density"); ax[2].set_title("3D")
for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e1_e_hist.png", dpi=140); plt.close(fig)

res.update(N=N, M=M)
json.dump(res, open("res/e1.json", "w"), indent=1)
print(res)
