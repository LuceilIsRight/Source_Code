"""Experiment 7: Elastic collisions of equal discs in a 2D box (time-stepped algorithm)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

rng = np.random.default_rng(7)
N, Lbox, R, m = 40, 10.0, 0.25, 1.0       # particles, box side, disc radius, mass
dt, Tend = 0.002, 20.0
nsteps = int(Tend / dt)

# initial condition: jittered 8 x 5 lattice (no overlaps), all speeds equal to 1, random directions
gx, gy = np.meshgrid((np.arange(8) + .5) * Lbox / 8, (np.arange(5) + .5) * Lbox / 5)
pos = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(-.3, .3, (N, 2))
ang = rng.uniform(0, 2 * np.pi, N)
vel = np.c_[np.cos(ang), np.sin(ang)]
vel -= vel.mean(axis=0)
vel *= np.sqrt(N / (vel**2).sum())        # fix kinetic energy: sum v^2 = N -> KE = N/2
I, J = np.triu_indices(N, 1)              # all particle pairs (i<j)

KE = lambda: 0.5 * m * (vel**2).sum()
snaps_t = [0, 1, 2, 5, 10, 20]
snaps, ts, ke = {}, [], []
speeds, ncoll = [], 0
for step in range(nsteps + 1):
    t = step * dt
    if step % 10 == 0:
        ts.append(t); ke.append(KE())
    if any(abs(t - s) < dt / 2 for s in snaps_t):
        snaps[round(t)] = pos.copy()
    if step % 100 == 0 and t >= 10:
        speeds.append(np.linalg.norm(vel, axis=1))
    # 1) free flight
    pos += vel * dt
    # 2) elastic reflection from the walls
    for k in range(2):
        lo = pos[:, k] < R;      pos[lo, k] = 2 * R - pos[lo, k];      vel[lo, k] *= -1
        hi = pos[:, k] > Lbox - R; pos[hi, k] = 2 * (Lbox - R) - pos[hi, k]; vel[hi, k] *= -1
    # 3) particle-particle collisions: detect overlap, swap normal velocity components
    d = pos[I] - pos[J]
    r2 = (d**2).sum(axis=1)
    for q in np.nonzero(r2 < (2 * R) ** 2)[0]:
        i, j = I[q], J[q]
        dx = pos[i] - pos[j]; r = np.linalg.norm(dx); n = dx / r
        vn = np.dot(vel[i] - vel[j], n)
        if vn < 0:                         # only if approaching
            vel[i] -= vn * n; vel[j] += vn * n; ncoll += 1
        shift = 0.5 * (2 * R - r) * n      # remove the overlap
        pos[i] += shift; pos[j] -= shift

ts, ke = np.array(ts), np.array(ke)
fig, ax = plt.subplots(2, 3, figsize=(11, 7))
for a, s in zip(ax.ravel(), snaps_t):
    for p in snaps[s]:
        a.add_patch(Circle(p, R, fc="tab:blue", ec="k", lw=.4, alpha=.7))
    a.set_xlim(0, Lbox); a.set_ylim(0, Lbox); a.set_aspect("equal"); a.set_title(f"t = {s}")
    a.set_xlabel("x"); a.set_ylabel("y")
fig.tight_layout(); fig.savefig("figs/e7_snaps.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(ts, ke); ax[0].set_ylim(0, 1.5 * ke[0])
ax[0].set_xlabel("time t"); ax[0].set_ylabel("total kinetic energy"); ax[0].set_title("Kinetic energy (conserved)"); ax[0].grid(alpha=.3)
sp = np.concatenate(speeds); Tk = ke[0] / N            # kT per particle in 2D
v = np.linspace(0, 3.5, 200)
ax[1].hist(sp, bins=20, density=True, alpha=.6, label="simulation, t = 10-20")
ax[1].plot(v, v / Tk * np.exp(-v**2 / (2 * Tk)), "k", label="2D Maxwell-Boltzmann")
ax[1].set_xlabel("speed |v|"); ax[1].set_ylabel("probability density"); ax[1].set_title("Speed distribution"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("figs/e7_ke.png", dpi=140); plt.close(fig)

res = dict(N=N, L=Lbox, R=R, dt=dt, Tend=Tend, KE0=float(ke[0]), KE_rel_dev=float(np.max(np.abs(ke - ke[0])) / ke[0]),
           collisions=int(ncoll), nsteps=nsteps)
json.dump(res, open("res/e7.json", "w"), indent=1)
print(res)
