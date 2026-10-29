"""Experiment 8: 3D molecular dynamics, Lennard-Jones particles, inelastic contacts, simulated annealing.
Reduced units: epsilon = sigma = m = k_B = 1."""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(8)
N, Lbox, dt = 64, 7.0, 0.004
rc = 2.5                                     # LJ cut-off (potential shifted to zero at rc)
Vshift = 4 * (rc**-12 - rc**-6)
e_p, e_w = 0.95, 0.95                        # restitution coefficients: particle-particle, particle-wall
r_hc = 0.9                                   # contact distance for inelastic particle collisions
tau = 0.5                                    # Berendsen thermostat time constant
T_init = 1.5

# initial condition: 4x4x4 simple cubic lattice, Maxwell velocities at T_init, zero total momentum
a0 = Lbox / 4
pos = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(4)]) * a0 + a0 / 2
vel = rng.normal(0, np.sqrt(T_init), (N, 3)); vel -= vel.mean(axis=0)
vel *= np.sqrt(3 * N * T_init / (vel**2).sum())
pos0 = pos.copy()

def forces(pos):
    """LJ forces and potential energy (shifted, cut at rc)."""
    d = pos[:, None, :] - pos[None, :, :]
    r2 = (d**2).sum(-1); np.fill_diagonal(r2, np.inf)
    mask = r2 < rc * rc
    inv2 = np.where(mask, 1.0 / r2, 0.0); inv6 = inv2**3
    ff = 24 * (2 * inv6**2 - inv6) * inv2            # F_vec = ff * d
    pe = 0.5 * np.sum(np.where(mask, 4 * (inv6**2 - inv6) - Vshift, 0.0))   # each pair once
    return (ff[:, :, None] * d).sum(1), pe

def walls():
    """Reflecting walls at 0 and Lbox with restitution e_w."""
    for k in range(3):
        lo = pos[:, k] < 0;    pos[lo, k] *= -1;               vel[lo, k] *= -e_w
        hi = pos[:, k] > Lbox; pos[hi, k] = 2 * Lbox - pos[hi, k]; vel[hi, k] *= -e_w

def contacts():
    """Inelastic collision when two particles come closer than r_hc while approaching."""
    d = pos[:, None, :] - pos[None, :, :]
    r2 = (d**2).sum(-1)
    for i, j in np.argwhere(np.triu(r2 < r_hc**2, 1)):
        n = d[i, j] / np.sqrt(r2[i, j]); vn = np.dot(vel[i] - vel[j], n)
        if vn < 0:
            vel[i] -= 0.5 * (1 + e_p) * vn * n; vel[j] += 0.5 * (1 + e_p) * vn * n

# annealing protocol: (name, target temperature (None = thermostat off), duration)
stages = [("Equilibration", 1.5, 20), ("Annealing step 1", 0.8, 25), ("Annealing step 2", 0.4, 25),
          ("Annealing step 3", 0.1, 30), ("Free relaxation", None, 20)]
F, pe = forces(pos)
t = 0.0; rec = {"t": [], "ke": [], "pe": []}; marks = []; stage_stats = []; snaps = {"initial": pos0.copy()}
for name, Tt, dur in stages:
    n = int(round(dur / dt)); marks.append((t, name, Tt)); kes, pes = [], []
    for s in range(n):
        vel += 0.5 * dt * F                       # velocity Verlet: half kick
        pos += dt * vel                           # drift
        walls(); contacts()
        F, pe = forces(pos)
        vel += 0.5 * dt * F                       # half kick
        ke = 0.5 * (vel**2).sum()
        if Tt is not None:                        # Berendsen velocity rescaling
            lam = np.sqrt(1 + (dt / tau) * (Tt / (2 * ke / (3 * N)) - 1)); vel *= lam; ke *= lam**2
        t += dt
        if s % 10 == 0:
            rec["t"].append(t); rec["ke"].append(ke); rec["pe"].append(pe)
        if s >= 0.8 * n and s % 10 == 0: kes.append(ke); pes.append(pe)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1); np.fill_diagonal(d, 9)
    stage_stats.append(dict(name=name, T_target=Tt, duration=dur, KE=float(np.mean(kes)), PE=float(np.mean(pes)),
                            E=float(np.mean(kes) + np.mean(pes)), T=float(2 * np.mean(kes) / (3 * N)),
                            PE_per_N=float(np.mean(pes) / N), coord=float((d < 1.5).sum() / N)))
    snaps[name] = pos.copy()

for k in rec: rec[k] = np.array(rec[k])
tt, ke, pe_, E = rec["t"], rec["ke"], rec["pe"], rec["ke"] + rec["pe"]
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(tt, ke, label="kinetic energy"); ax.plot(tt, pe_, label="potential energy"); ax.plot(tt, E, "k", lw=1, label="total energy")
for (t0, name, Tt) in marks:
    ax.axvline(t0, color="gray", ls=":", lw=.8)
    ax.text(t0 + .6, 195, name.replace("Annealing step", "Anneal"), rotation=0, fontsize=7, va="top")
ax.set_ylim(-310, 200); ax.set_xlabel("time (reduced units)"); ax.set_ylabel(r"energy ($\epsilon$)"); ax.set_title("Energy during the annealing protocol")
ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e8_energy.png", dpi=140); plt.close(fig)

fig = plt.figure(figsize=(11, 3.8))
for k, key in enumerate(("initial", "Equilibration", "Annealing step 3", "Free relaxation")):
    a = fig.add_subplot(1, 4, k + 1, projection="3d")
    p = snaps[key]; a.scatter(p[:, 0], p[:, 1], p[:, 2], s=35, c=p[:, 2], cmap="viridis", edgecolor="k", lw=.3)
    a.set_xlim(0, Lbox); a.set_ylim(0, Lbox); a.set_zlim(0, Lbox)
    a.set_title({"initial": "initial lattice (t=0)", "Equilibration": "end of equilibration", "Annealing step 3": "end of anneal step 3",
                 "Free relaxation": "final state"}[key], fontsize=8)
    a.set_xlabel("x", fontsize=7); a.set_ylabel("y", fontsize=7); a.set_zlabel("z", fontsize=7); a.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig("figs/e8_snaps.png", dpi=140); plt.close(fig)

res = dict(N=N, L=Lbox, dt=dt, rc=rc, e_p=e_p, e_w=e_w, r_hc=r_hc, tau=tau, T_init=T_init, stages=stage_stats,
           steps=int(sum(round(s[2] / dt) for s in stages)))
json.dump(res, open("res/e8.json", "w"), indent=1)
for s in stage_stats: print(s)
