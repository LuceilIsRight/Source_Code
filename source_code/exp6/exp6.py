"""Experiment 6: Free fall - Euler, position Verlet and velocity Verlet versus the exact solution."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

g, y0, v0 = 9.81, 100.0, 0.0           # gravity (m/s^2), initial height (m), initial velocity (m/s)
a = lambda y: -g                        # acceleration (constant, independent of position)

def euler(dt, T):
    n = int(round(T / dt)); y = np.empty(n + 1); v = np.empty(n + 1); y[0], v[0] = y0, v0
    for i in range(n):
        y[i + 1] = y[i] + v[i] * dt              # position uses OLD velocity
        v[i + 1] = v[i] + a(y[i]) * dt
    return y

def position_verlet(dt, T):
    """Drift-kick-drift form: half drift, kick with a at midpoint, half drift."""
    n = int(round(T / dt)); y = np.empty(n + 1); v = v0; y[0] = y0
    for i in range(n):
        ym = y[i] + 0.5 * dt * v
        v = v + a(ym) * dt
        y[i + 1] = ym + 0.5 * dt * v
    return y

def velocity_verlet(dt, T):
    """y += v dt + a dt^2/2;  v += (a_old + a_new) dt/2."""
    n = int(round(T / dt)); y = np.empty(n + 1); v = v0; y[0] = y0
    for i in range(n):
        a_old = a(y[i])
        y[i + 1] = y[i] + v * dt + 0.5 * a_old * dt**2
        v = v + 0.5 * (a_old + a(y[i + 1])) * dt
    return y

exact = lambda t: y0 + v0 * t - 0.5 * g * t**2
T = 4.0                                  # final time (ground reached at t = 4.515 s)
methods = {"Euler": euler, "Position Verlet": position_verlet, "Velocity Verlet": velocity_verlet}

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, dt in enumerate((0.5, 0.1)):
    tt = np.linspace(0, T, 400)
    ax[k].plot(tt, exact(tt), "k-", lw=1, label="analytic")
    t = np.arange(int(round(T / dt)) + 1) * dt
    for name, mk in zip(methods, ("o", "s", "^")):
        ax[k].plot(t, methods[name](dt, T), mk, ms=5 if dt > .2 else 3, mfc="none", label=name)
    ax[k].set_xlabel("time t (s)"); ax[k].set_ylabel("height y (m)")
    ax[k].set_title(f"Free fall, $\\Delta t$ = {dt} s"); ax[k].legend(fontsize=8); ax[k].grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e6_height.png", dpi=140); plt.close(fig)

# error versus time, dt = 0.5
dt = 0.5; t = np.arange(int(round(T / dt)) + 1) * dt
fig, ax = plt.subplots(figsize=(6, 3.8))
for name in methods:
    ax.semilogy(t, np.abs(methods[name](dt, T) - exact(t)) + 1e-16, "o-", ms=4, label=name)
ax.set_xlabel("time t (s)"); ax.set_ylabel("|y_num - y_exact| (m)")
ax.set_title("Absolute error, $\\Delta t$ = 0.5 s"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig("figs/e6_error_t.png", dpi=140); plt.close(fig)

# convergence: error at t = T versus time step
dts = [0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]
err = {name: [abs(methods[name](d, T)[-1] - exact(T)) for d in dts] for name in methods}
fig, ax = plt.subplots(figsize=(6, 4))
for name in methods:
    ax.loglog(dts, np.maximum(err[name], 1e-15), "o-", label=name)
ax.loglog(dts, [g * d * T / 2 for d in dts], "k--", lw=.8, label=r"$g\,\Delta t\,t/2$")
ax.set_xlabel(r"time step $\Delta t$ (s)"); ax.set_ylabel("error in height at t = 4 s (m)")
ax.set_title("Convergence with time step"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig("figs/e6_conv.png", dpi=140); plt.close(fig)
slope = float(np.polyfit(np.log(dts), np.log(err["Euler"]), 1)[0])
res = dict(g=g, y0=y0, v0=v0, T=T, dts=dts, err={k: [float(x) for x in v] for k, v in err.items()}, euler_slope=slope,
           t_ground=float(np.sqrt(2 * y0 / g)))
json.dump(res, open("res/e6.json", "w"), indent=1)
print(res)
