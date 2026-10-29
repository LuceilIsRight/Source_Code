"""Experiment 5: Large-angle pendulum, RK4 integration (no small-angle approximation)."""
import json
import numpy as np
from scipy.special import ellipk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

g, L = 9.81, 1.0
dt = 1.0e-3
T0 = 2 * np.pi * np.sqrt(L / g)               # small-angle period

def f(phi, om):
    """phi' = omega, omega' = -(g/L) sin(phi)."""
    return om, -(g / L) * np.sin(phi)

def rk4(phi, om, dt):
    """RK4 step (works on arrays, so many initial angles run simultaneously)."""
    a1, b1 = f(phi, om)
    a2, b2 = f(phi + .5 * dt * a1, om + .5 * dt * b1)
    a3, b3 = f(phi + .5 * dt * a2, om + .5 * dt * b2)
    a4, b4 = f(phi + dt * a3, om + dt * b3)
    return (phi + dt * (a1 + 2 * a2 + 2 * a3 + a4) / 6,
            om + dt * (b1 + 2 * b2 + 2 * b3 + b4) / 6)

def integrate(phi0, tmax):
    """Integrate from rest at phi0; return time array and phi(t), omega(t) histories."""
    n = int(round(tmax / dt))
    phi = np.array(phi0, float); om = np.zeros_like(phi)
    P = np.empty((n + 1,) + phi.shape); W = np.empty_like(P)
    P[0], W[0] = phi, om
    for i in range(n):
        phi, om = rk4(phi, om, dt)
        P[i + 1], W[i + 1] = phi, om
    return np.arange(n + 1) * dt, P, W

# (b) phase-space plots
angles_deg = np.array([10, 30, 60, 90, 120, 150, 170])
t, P, W = integrate(np.radians(angles_deg), 12.0)
fig, ax = plt.subplots(figsize=(6, 4.6))
for k, a in enumerate(angles_deg):
    ax.plot(P[:, k], W[:, k], lw=.9, label=f"$\\phi_0$={a}$^\\circ$")
ax.set_xlabel(r"angular displacement $\phi$ (rad)"); ax.set_ylabel(r"angular velocity $d\phi/dt$ (rad/s)")
ax.set_title("Pendulum phase space"); ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e5_phase.png", dpi=140); plt.close(fig)

# time series example
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for k in (0, 3, 6):
    ax.plot(t, P[:, k], label=f"$\\phi_0$={angles_deg[k]}$^\\circ$")
ax.set_xlabel("time t (s)"); ax.set_ylabel(r"$\phi$ (rad)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("Pendulum angle versus time")
fig.tight_layout(); fig.savefig("figs/e5_time.png", dpi=140); plt.close(fig)

# energy conservation check (RK4 accuracy), E/m L^2 = om^2/2 + (g/L)(1-cos phi)
E = 0.5 * W**2 + (g / L) * (1 - np.cos(P))
edrift = float(np.max(np.abs(E - E[0]) / E[0]))

# (c) period versus amplitude: T = 2*(t_cross2 - t_cross1) (phi crosses 0 at T/4 and 3T/4)
deg = np.arange(2, 176, 2.0)
t, P, W = integrate(np.radians(deg), 14.0)
T_num = np.empty(deg.size)
for k in range(deg.size):
    s = np.where(P[:-1, k] * P[1:, k] < 0)[0]
    tc = [t[i] + dt * P[i, k] / (P[i, k] - P[i + 1, k]) for i in s[:2]]     # linear interpolation
    T_num[k] = 2 * (tc[1] - tc[0])
T_exact = 4 * np.sqrt(L / g) * ellipk(np.sin(np.radians(deg) / 2) ** 2)     # elliptic-integral result
T_series = T0 * (1 + np.radians(deg) ** 2 / 16)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(deg, T_num, "o", ms=3.5, label="numerical (RK4)")
ax.plot(deg, T_exact, "k-", lw=.8, label="exact (elliptic integral)")
ax.plot(deg, T_series, "g:", label=r"$T_0(1+\phi_0^2/16)$")
ax.axhline(T0, color="r", ls="--", label=fr"small-angle $T_0={T0:.3f}$ s")
ax.set_xlabel(r"initial angle $\phi_0$ (degrees)"); ax.set_ylabel("period T (s)")
ax.set_title("Period versus amplitude"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/e5_period.png", dpi=140); plt.close(fig)

rows = []
for a in (5, 10, 30, 60, 90, 120, 150, 170):
    k = int(np.argmin(np.abs(deg - a)))
    if abs(deg[k] - a) > 1e-9: k = int(np.argmin(np.abs(deg - (a + 1))))
    rows.append((float(deg[k]), float(T_num[k]), float(T_exact[k]), float(T_num[k] / T0)))
res = dict(T0=float(T0), rows=rows, max_rel_err=float(np.max(np.abs(T_num / T_exact - 1))),
           energy_drift=edrift, dt=dt, g=g, L=L)
json.dump(res, open("res/e5.json", "w"), indent=1)
print(res)
