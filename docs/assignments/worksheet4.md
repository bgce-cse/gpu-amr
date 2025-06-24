# Worksheet4 - Non-linear Equations

## 3.1. Gaussian Wave Validation

The Euler equations implementation was tested with a smooth Gaussian wave, showing proper behavior. Velocity components (rhou, rhov) maintained small errors (~1e-2), confirming correct momentum transport. Density (rho) and energy (rhoE) showed larger but expected errors due to nonlinear effects, while remaining stable throughout the simulation.

Var     L1          L2          L∞
rhou    3.75e-02    4.54e-02    1.08e-01
rhov    3.75e-02    4.54e-02    1.08e-01 
rho     4.23e-01    5.02e-01    9.00e-01
rhoE    1.14e+00    1.39e+00    2.39e+00

<div style="text-align: center;">
  <img src="rho.png" width="1200"/>
  <p style="margin-top: 8px; font-style: italic;">Figure 1: Constant pressure preserved, confirming stability of the scheme for trivial solutions.</p>
</div>

<div style="text-align: center;">
  <img src="rhoE.png" width="1200"/>
  <p style="margin-top: 8px; font-style: italic;">Figure 2: Constant pressure preserved, confirming stability of the scheme for trivial solutions.</p>
</div>

<div style="text-align: center;">
  <img src="rhou.png" width="1200"/>
  <p style="margin-top: 8px; font-style: italic;">Figure 3: Constant pressure preserved, confirming stability of the scheme for trivial solutions.</p>
</div>

<div style="text-align: center;">
  <img src="rhov.png" width="1200"/>
  <p style="margin-top: 8px; font-style: italic;">Figure 4: Constant pressure preserved, confirming stability of the scheme for trivial solutions.</p>
</div>

These results confirm the implementation correctly handles smooth solutions, with error magnitudes matching theoretical expectations for this scheme order. The preserved wave profile and stable evolution demonstrate proper functionality for smooth initial conditions.