"""Experiment 2: Lotka-Volterra predator-prey model solved with RK4."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Model parameters (arbitrary time units)
alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5   # prey growth, predation, predator gain, predator death

def rhs(s, a=alpha, b=beta, dl=delta, g=gamma):
    """Right-hand side: s = (prey x, predator y)."""
    x, y = s
    return np.array([a * x - b * x * y, dl * x * y - g * y])

def rk4_step(s, dt, **p):
    """One classical 4th-order Runge-Kutta step."""
    k1 = rhs(s, **p)
    k2 = rhs(s + 0.5 * dt * k1, **p)
    k3 = rhs(s + 0.5 * dt * k2, **p)
    k4 = rhs(s + dt * k3, **p)
    return s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

def solve(s0, T=50.0, dt=0.005, **p):
    n = int(round(T / dt))
    out = np.empty((n + 1, 2)); out[0] = s0
    for i in range(n):
        out[i + 1] = rk4_step(out[i], dt, **p)
    return np.linspace(0, T, n + 1), out

def invariant(s, a=alpha, b=beta, dl=delta, g=gamma):
    """Conserved quantity H = dl*x - g*ln x + b*y - a*ln y."""
    return dl * s[:, 0] - g * np.log(s[:, 0]) + b * s[:, 1] - a * np.log(s[:, 1])

res = {}
xe, ye = gamma / delta, alpha / beta               # equilibrium point
res["equilibrium"] = [xe, ye]

# (b) time evolution for the baseline run
t, s = solve([40.0, 9.0])
H = invariant(s)
res["H_rel_drift"] = float(np.max(np.abs(H - H[0])) / abs(H[0]))
pk = np.where((s[1:-1, 0] > s[:-2, 0]) & (s[1:-1, 0] > s[2:, 0]))[0] + 1   # prey maxima
res["period_baseline"] = float(np.mean(np.diff(t[pk])))
res["prey_max"], res["prey_min"] = float(s[:, 0].max()), float(s[:, 0].min())
res["pred_max"], res["pred_min"] = float(s[:, 1].max()), float(s[:, 1].min())
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(t, s[:, 0], label="prey x(t)"); ax.plot(t, s[:, 1], label="predator y(t)")
ax.set_xlabel("time t"); ax.set_ylabel("population (individuals)")
ax.set_title("Predator-prey populations, $(x_0,y_0)=(40,9)$"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e2_time.png", dpi=140); plt.close(fig)

# (c) phase space for several initial conditions
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for ic in ([20, 10], [25, 12], [30, 12], [40, 9], [60, 10]):
    _, q = solve(ic, T=40)
    ax.plot(q[:, 0], q[:, 1], label=f"({ic[0]},{ic[1]})")
ax.plot(xe, ye, "k*", ms=11, label="equilibrium (20,10)")
ax.set_xlabel("prey population x"); ax.set_ylabel("predator population y")
ax.set_title("Phase space for different initial conditions"); ax.legend(fontsize=8, title="$(x_0,y_0)$"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e2_phase.png", dpi=140); plt.close(fig)

# (d) effect of parameters: vary alpha and delta (same initial condition)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a_ in (0.5, 1.0, 1.5):
    tt, q = solve([40, 9], T=40, a=a_)
    ax[0].plot(q[:, 0], q[:, 1], label=f"alpha={a_}")
    ax[0].plot(gamma / delta, a_ / beta, "k*")
for d_ in (0.05, 0.075, 0.1):
    tt, q = solve([40, 9], T=40, dl=d_)
    ax[1].plot(q[:, 0], q[:, 1], label=f"delta={d_}")
    ax[1].plot(gamma / d_, alpha / beta, "k*")
for a, ttl in zip(ax, ("varying prey growth rate alpha", "varying predator efficiency delta")):
    a.set_xlabel("prey population x"); a.set_ylabel("predator population y"); a.set_title(ttl)
    a.legend(fontsize=8); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e2_params.png", dpi=140); plt.close(fig)

# period versus amplitude (initial condition (x0, 10)); small oscillation limit 2*pi/sqrt(alpha*gamma)
res["period_small_theory"] = float(2 * np.pi / np.sqrt(alpha * gamma))
per = {}
for x0 in (22, 40, 70):
    tt, q = solve([x0, 10.0], T=80)
    p_ = np.where((q[1:-1, 0] > q[:-2, 0]) & (q[1:-1, 0] > q[2:, 0]))[0] + 1
    per[x0] = float(np.mean(np.diff(tt[p_])))
res["period_vs_x0"] = per
json.dump(res, open("res/e2.json", "w"), indent=1)
print(res)
