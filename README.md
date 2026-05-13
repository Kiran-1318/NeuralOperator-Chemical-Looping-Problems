# NeuralOperator Chemical Looping Problems

A series of 5 neural operator problems building toward a physics-informed
surrogate for Fe₂O₃ redox kinetics in chemical looping gasification.

All problems are original formulations. Each builds one specific concept
needed to bridge from single-condition PINNs to operators that generalise
across the full temperature parameter space.

## Companion repository

This series follows
[DeepXDE-Chemical-Looping-Problems](https://github.com/Kiran-1318/DeepXDE-Chemical-Looping-Problems)
— 10 original PINN problems covering forward, inverse, and hybrid methods.

## Problem series

### Tier 1 — Data-driven operators (Problems 1–3)
Operator learning without physics constraints.

| # | Problem | Operator learned | Key skill | Result |
|---|---------|-----------------|-----------|--------|
| 1 | DeepONet: batch reactor | k → C(t) | Branch + trunk architecture, dot product output | All 5 unseen k values < 1% Rel L2 |
| 2 | DeepONet: shrinking core | k → X(t) | Numerical data generation with solve_ivp, sigmoid output constraint | All 10 unseen k values < 2% Rel L2 |
| 3 | DeepONet: temperature-parametric | T → X(t) | Arrhenius operator, per-temperature t_max | Validated at T=750K |

Problems 1 and 2 include both DeepXDE and raw PyTorch from-scratch implementations.

### Tier 2 — Advanced architectures (Problem 4)

| # | Problem | Operator learned | Key skill | Result |
|---|---------|-----------------|-----------|--------|
| 4 | FNO: non-isothermal coupled ODE | [C₀, T₀, cool_coef] → [C(t), T(t)] | Fourier Neural Operator from scratch — SpectralConv1d, GELU dual-path layers | C Rel L2: 0.12%, T Rel L2: 0.06% |

FNO built entirely from scratch in pure PyTorch — no library used.

### Tier 3 — Physics-informed operator (Problem 5)
Most important problem. Direct structural template for the Fe₂O₃ preprint.

| # | Problem | Operator learned | Key skill | Result |
|---|---------|-----------------|-----------|--------|
| 5 | PI-DeepONet: Fe₂O₃ surrogate | T → X(t) with physics constraint | Physics residual loss + sparse TGA data, generalisation to unseen temperatures | See table below |

**Key comparison — unseen temperature generalisation:**

| Method | T=700K | T=800K | T=900K | Physics enforced |
|--------|--------|--------|--------|-----------------|
| Data-only DeepONet | 36.03% | 37.19% | 35.16% | No |
| PI-DeepONet | 1.22% | 1.04% | 0.36% | Yes |
| Problem 10 PINN (single T) | ~1.57% | retrain needed | retrain needed | Yes |

Physics constraints enable **30× better generalisation** to unseen temperatures.
Trained on only 24 sparse noisy observations at 3 temperatures.

## Key technical lessons

**DeepONet (Problems 1–3):**
- Branch net encodes input function (k or T) → p-dimensional vector
- Trunk net encodes query location (t) → p-dimensional vector
- Output = dot product + bias: `torch.mm(b, τ.T) + bias`
- Resolution fixed at training grid — trunk points must match training
- Best suited for scalar parameter inputs

**FNO (Problem 4):**
- `SpectralConv1d`: FFT → multiply complex weights → IFFT
- `einsum('bim, iom -> bom')` — mixes input channels per frequency mode
- `n_modes=16` — keeps low-frequency components, discards high-frequency noise
- Two parallel paths per layer: spectral (global) + pointwise (local)
- Resolution-independent at inference — unlike DeepONet
- Best suited for function-to-function mappings on grids

**Physics-informed DeepONet (Problem 5):**
- Physics residual loss enforced at collocation points across full (T, t) domain
- Per-temperature `t_max = (3/k) × 0.95` for informative collocation points
- IC loss `X(T, t=0) = 0` enforced for all T ∈ [600, 900] K
- Chain rule for physical time derivative: `dX/dt_phys = dX/dt_norm / t_max_T`
- Data loss with `lambda_data` weight — check epoch 0 magnitudes before setting

## Research context

This work precedes a Physics-Informed Neural Network for Fe₂O₃ reduction
kinetics in chemical looping gasification for hydrogen production,
targeting a ChemRxiv preprint. The PI-DeepONet from
Problem 5 is the direct comparison method in the preprint.

**Connection to Fe₂O₃ preprint (Section 4):**

| Problem 5 | Fe₂O₃ paper |
|-----------|-------------|
| Shrinking core ODE | Fe₂O₃ reduction kinetics |
| 24 sparse observations at 3 T values | TGA experimental data |
| Data-only DeepONet | ML baseline |
| PI-DeepONet | Operator learning contribution |
| T=700K, 800K unseen validation | Generalisation to new TGA conditions |

## Implementation

- **Framework:** DeepXDE (Problems 1–3 DeepXDE versions) + pure PyTorch
- **Problem 1:** DeepXDE + raw PyTorch
- **Problem 2:** DeepXDE + raw PyTorch
- **Problems 3–5:** raw PyTorch only

## Related repositories

[1D-Heat-Equation-PINN](https://github.com/Kiran-1318/1D-Heat-Equation-PINN)
— Physics-Informed Neural Network for the 1D transient heat equation.
Relative L2 error: 0.99%.

[DeepXDE-Chemical-Looping-Problems](https://github.com/Kiran-1318/DeepXDE-Chemical-Looping-Problems)
— 10 original PINN problems: forward ODEs, inverse parameter identification,
multi-cycle redox kinetics, and physics-data hybrid PINN (preprint template).

## Author

**Kiran Thammina**
M.Tech Energy Systems Engineering, IIT Bombay (CPI 9.84, Best Thesis Award)
GitHub: [github.com/Kiran-1318](https://github.com/Kiran-1318)
