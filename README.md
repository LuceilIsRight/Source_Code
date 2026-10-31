# Computational Physics II: Lab Journal and CCA-II

Python source code for the Computational Physics II laboratory (eight numerical experiments) and the CCA-II assessment (1D diffusion equation solved with the Crank-Nicolson method), together with the scripts that assemble the written reports as PDF and Word files.

Every script uses a fixed random seed, so all figures and numbers in the reports can be reproduced exactly.

## Contents
Script	Topic	Main method
exp1.py	Random walks in 1D, 2D and 3D	Unit-step walks; mean displacement, mean square displacement and histograms over 2000 walks
exp2.py	Predator-prey (Lotka-Volterra) model	RK4; time series, phase portraits, parameter study, conserved-quantity check
exp3.py	Area of a circle	Hit-or-miss Monte Carlo; N^(-1/2) error scaling
exp4.py	Gaussian integrals in 1D, 2D, 3D	Uniform-sampling Monte Carlo compared with the analytic result
exp5.py	Large-angle pendulum	RK4; phase space and period versus amplitude, compared with the elliptic-integral period
exp6.py	Free fall: integrator comparison	Euler, position Verlet and velocity Verlet against the analytic solution
exp7.py	Elastic collisions in a 2D box	Time-stepped hard-disc simulation; snapshots, kinetic energy, speed distribution
exp8.py	3D molecular dynamics and simulated annealing	Lennard-Jones, velocity Verlet, inelastic collisions, Berendsen thermostat cooling in three steps
cca2.py	1D diffusion equation (CCA-II)	Crank-Nicolson with a Thomas tridiagonal solver; validated against the exact Fourier series

## Report builders:

Script	Output
build.py	Computational_Physics_Lab_Journal.pdf (ReportLab)
build_cca2.py	CCA-II.pdf (ReportLab)
build_docx.py + render_docx.js	Computational_Physics_Lab_Journal.docx
build_cca2_docx.py + render_docx.js	CCA-II.docx

The report builders read the figures in figs/ and the numerical results in res/, so run the experiment scripts first.

## Requirements
Python 3.10 or newer (developed with 3.12)
Python packages: numpy, scipy, matplotlib, reportlab, Pillow
Optional, for the Word files only: Node.js and the docx package
bash
pip install -r requirements.txt
npm install            # optional: Word output
Quick start

## Run everything (experiments, PDF reports and, if npm install was done, Word reports):

bash
./run_all.sh

This takes about a minute and a half on an ordinary laptop; exp8.py (the molecular-dynamics run) is the slowest part.

To run a single experiment, create the output folders first, because the scripts write there:

bash
mkdir -p figs res eq
python3 exp5.py          # writes figs/e5_*.png and res/e5.json
python3 cca2.py          # writes figs/cca_*.png and res/cca.json
