"""Dirichlet boundaries u(0,t) = u(L,t) = 0, initial condition u(x,0) = exp(-((x - L/2)/0.1)^2)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- numerical parameters ----------------
L = 1.0                          # domain length
D = 0.01                         # diffusion coefficient
N = 201                          # number of grid points (including the two boundary points)
dt = 0.01                        # time step
t_out = [0.0, 1.0, 5.0, 10.0]    # output times

def initial(x, L):
    """Initial condition u(x,0) = exp[-((x - L/2)/0.1)^2]."""
    return np.exp(-((x - L / 2) / 0.1) ** 2)

def tridiag_factor(a, b, c, m):
    """Factorise the constant tridiagonal matrix tridiag(a, b, c) of size m (Thomas algorithm).
    Because A does not change in time, this is done only once."""
    cp = np.empty(m); den = np.empty(m)
    den[0] = b; cp[0] = c / b
    for i in range(1, m):
        den[i] = b - a * cp[i - 1]
        cp[i] = c / den[i]
    return cp, den

def tridiag_solve(a, cp, den, d):
    """Solve A x = d using the stored factorisation: forward elimination, then back substitution."""
    m = d.size; y = np.empty(m); x = np.empty(m)
    y[0] = d[0] / den[0]
    for i in range(1, m):
        y[i] = (d[i] - a * y[i - 1]) / den[i]
    x[-1] = y[-1]
    for i in range(m - 2, -1, -1):
        x[i] = y[i] - cp[i] * x[i + 1]
    return x

def crank_nicolson(L, D, N, dt, t_out):
    """Advance the solution with Crank-Nicolson; return x and a dict {t: u(x,t)} for each output time."""
    x = np.linspace(0.0, L, N)                   # (a) uniform grid, dx = L/(N-1)
    dx = x[1] - x[0]
    r = D * dt / dx**2
    u = initial(x, L)                            # (b) initial condition
    u[0] = u[-1] = 0.0                           # (c) boundary conditions
    m = N - 2                                    # number of interior unknowns
    a = c = -0.5 * r; b = 1.0 + r                # A = tridiag(-r/2, 1+r, -r/2)
    cp, den = tridiag_factor(a, b, c, m)
    steps_out = {int(round(t / dt)): t for t in t_out}
    out = {}
    if 0 in steps_out: out[0.0] = u.copy()
    for n in range(1, int(round(max(t_out) / dt)) + 1):
        # right-hand side B u^n, B = tridiag(r/2, 1-r, r/2)
        rhs = 0.5 * r * (u[:-2] + u[2:]) + (1.0 - r) * u[1:-1]
        u[1:-1] = tridiag_solve(a, cp, den, rhs)  # (d) solve A u^{n+1} = B u^n
        u[0] = u[-1] = 0.0                       # (c) impose boundary conditions every step
        if n in steps_out: out[steps_out[n]] = u.copy()   # (e) store output
    return x, out, r

def exact(x, t, L=L, D=D, nmax=600):
    """Exact solution by a Fourier sine series (used only to validate the numerical result)."""
    xf = np.linspace(0, L, 40001)
    n = np.arange(1, nmax + 1)[:, None]
    bn = 2.0 / L * np.trapezoid(initial(xf, L) * np.sin(n * np.pi * xf / L), xf, axis=1)
    return (bn[:, None] * np.sin(n * np.pi * x / L) * np.exp(-D * (n * np.pi / L) ** 2 * t)).sum(axis=0)

if __name__ == "__main__":
    x, sol, r = crank_nicolson(L, D, N, dt, t_out)
    dx = x[1] - x[0]
    # ---- required plot ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in t_out:
        ax.plot(x, sol[t], lw=1.8, label=f"t = {t:g}")
    ax.set_xlabel("position x"); ax.set_ylabel("u(x, t)")
    ax.set_title(f"1D diffusion, Crank-Nicolson (L={L:g}, D={D:g}, N={N}, $\\Delta t$={dt:g})")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("figs/cca_main.png", dpi=150); plt.close(fig)

    # ---- validation against the exact series ----
    err = {t: float(np.max(np.abs(sol[t] - (initial(x, L) if t == 0 else exact(x, t))))) for t in t_out}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, t in enumerate(t_out):
        ex = initial(x, L) if t == 0 else exact(x, t)
        ax[0].plot(x, ex, "-", color=f"C{k}", lw=1.2, label=f"exact, t={t:g}")
        ax[0].plot(x[::8], sol[t][::8], "o", color=f"C{k}", ms=3.5, mfc="none")
        ax[1].semilogy(x, np.abs(sol[t] - ex) + 1e-18, color=f"C{k}", label=f"t = {t:g}")
    ax[0].plot([], [], "ko", mfc="none", ms=3.5, label="Crank-Nicolson")
    ax[0].set_xlabel("position x"); ax[0].set_ylabel("u(x, t)"); ax[0].set_title("Numerical vs exact solution"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("position x"); ax[1].set_ylabel("|u_num - u_exact|"); ax[1].set_title("Absolute error"); ax[1].legend(fontsize=7)
    for a_ in ax: a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("figs/cca_valid.png", dpi=150); plt.close(fig)

    # ---- diagnostics: peak value, area, free-space comparison ----
    t0 = (0.1) ** 2 / (4 * D)                    # initial Gaussian = free-space solution at t0
    diag = {str(t): dict(peak=float(sol[t].max()), area=float(np.trapezoid(sol[t], x)),
                         free_peak=float(np.sqrt(t0 / (t0 + t))), maxerr=err[t],
                         exact_peak=float((initial(x, L) if t == 0 else exact(x, t)).max())) for t in t_out}
    # ---- sensitivity to dt (N fixed) and to dx (dt small), error at t = 1 ----
    ex1 = exact(x, 1.0)
    dt_tab = []
    for d_ in (0.5, 0.1, 0.01, 0.001):
        xx, s, rr = crank_nicolson(L, D, N, d_, [1.0]); dt_tab.append((d_, float(rr), float(np.max(np.abs(s[1.0] - ex1)))))
    dx_tab = []
    for n_ in (51, 101, 201, 401):
        xx, s, rr = crank_nicolson(L, D, n_, 0.001, [1.0])
        dx_tab.append((n_, float(xx[1] - xx[0]), float(np.max(np.abs(s[1.0] - exact(xx, 1.0))))))
    res = dict(L=L, D=D, N=N, dt=dt, dx=float(dx), r=float(r), diag=diag, dt_tab=dt_tab, dx_tab=dx_tab,
               u0_at_boundary=float(initial(np.array([0.0]), L)[0]), steps=int(round(10 / dt)),
               dt_explicit=float(0.5 * dx**2 / D))
    json.dump(res, open("res/cca.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
