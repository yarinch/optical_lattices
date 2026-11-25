# optical_lattices

[Yarin Cohen](https://github.com/yarinch) · Max Planck Institute of Quantum Optics

Ab-initio calculations for tunable two-dimensional optical lattices, directly from lab parameters: band structures,
maximally localized Wannier functions, tight-binding hoppings, Hubbard $U$, and time evolution under ramps of the
lattice.

The lattice is the programmable optical lattice of D. Wei *et al.*,
[Phys. Rev. X **13**, 021042 (2023)](https://doi.org/10.1103/PhysRevX.13.021042): a one-dimensional lattice $L_1$
along $x = y$ (a counter-propagating beam, depth $V_1$) on top of a square lattice $L_2$ (a bowtie configuration,
depth $V_2$), with a relative phase $\phi$:

$$V(x, y) = -V_1 \cos^2\big(k(x+y) + \phi/2\big) - V_2 \cos^2(kx)\cos^2(ky).$$

Tuning $V_1/V_2$ and $\phi$ gives, among others, the brick-wall lattice ($\phi = \pi$, a honeycomb-like lattice of
double wells with Dirac points) and the triangular lattice ($\phi = 0$).

## Installation

Requires Python ≥ 3.10.

```bash
pip install git+https://github.com/yarinch/optical_lattices.git
```

The Wannier functions are computed with [Wannier90](https://wannier.org) (tested with 3.1.0), which has to be installed
separately. The package looks for the binary `wannier90.x` in this order:

1. `Config(wannier90_dir=...)`, the directory containing it;
2. the environment variable `WANNIER90_X`, the full path to the binary;
3. `wannier90.x` on the `PATH`.

On Windows, `Config(wannier90_executor="WSL")` runs Wannier90 inside WSL instead.

## Quick start

```python
import numpy as np
from optical_lattices import Config, Potential, Energy, Wannier

config = Config(V1=15, V2=11, phi=np.pi)          # brick-wall lattice, depths in E_r

# Potential and bands
V = Potential(config).potential(0.2, 0.2)
bands = Energy(config).get_energies(0.0, 0.0, num_bands=4)

# Wannier functions and tight-binding model: initial guesses near the two wells B = (+, +), A = (-, -)
wannier = Wannier(config, [(0.2, 0.2), (-0.2, -0.2)])
result = wannier.wannierize()

H = result.hamiltonian                            # {(R, m, n): <w_m(0)|H|w_n(R)>}, in E_r
t1, t2 = -H[((0, 0), 1, 2)], -H[((1, 0), 1, 2)]   # hoppings within and between unit cells
U = wannier.hubbard_U(result)                     # on-site interaction (see Config.w_z)
```

Time evolution, e.g. splitting a single well into a double well and preparing $(|w_B\rangle + |w_A\rangle)/\sqrt 2$:

```python
from optical_lattices import Wavefunction, LinearRamp, Evolution

start = Config(V1=0, V2=16, phi=np.pi, num_wannier=1, num_q=12, L_max=8)
end = Config(V1=20, V2=16, phi=np.pi, num_wannier=2, num_q=12, L_max=8)

psi0 = Wavefunction.from_wannier(Wannier(start, [(0, 0)]).wannierize(), [1])
target = Wavefunction.from_wannier(Wannier(end, [(0.2, 0.2), (-0.2, -0.2)]).wannierize(), np.array([1, 1]) / np.sqrt(2))

result = Evolution(LinearRamp(start, end, duration=1.0)).run(psi0, dt=0.005)
result.final.fidelity(target)                     # ~0.96
```

[`examples/01_usage.ipynb`](examples/01_usage.ipynb) walks through all of this with plots.

## Modules

| Class | What it does |
|---|---|
| `Config` | Lattice depths and phase, numerical parameters, atomic parameters. Every module takes a `Config`. |
| `Potential` | The potential $V(x, y)$ in real space. |
| `Energy` | Band structure from a plane-wave expansion. |
| `Wannier` | Maximally localized Wannier functions (via Wannier90), the tight-binding Hamiltonian, the Hubbard $U$. |
| `Wavefunction`, `Ramp`, `LinearRamp`, `Evolution` | Time evolution of states during a ramp of the lattice (split-operator method). |
| `TwistedBilayer` | Tight-binding model of two twisted brick-wall layers (moiré lattices). |

## Conventions

- **Units.** $\hbar = 1$, lengths in the lattice constant $a = \pi/k = 1$, energies in recoil units
  $E_r = \hbar^2k^2/2m = 1$, times in $\hbar/E_r$.
- **Depths.** $V_2$ is the depth of the resulting square lattice, not of the generating laser beam (which is $V_2/4$).
- **Tight binding.** `result.hamiltonian[(R, m, n)]` $= \langle w_m(0)|H|w_n(R)\rangle$, with $R$ the lattice vector in
  units of $a$ and $m, n$ the Wannier indices in the order of the initial guesses (starting at 1). Hoppings are
  $t = -H$.
- **Hubbard U.** Uses the scattering length `a_s_bohr` (default 95 $a_0$, <sup>87</sup>Rb) and the confinement along $z$,
  `w_z` $= \hbar\omega_z$ in $E_r$; $U \propto \sqrt{\omega_z}$, and the default `w_z = 1` gives $U$ per $\sqrt{\omega_z}$.
- **Numerics.** `L_max` sets the plane-wave cutoff, `num_q` the k-grid for Wannier90 (and the system size, a torus of
  `num_q` $\times$ `num_q` cells, for the time evolution). The `num_wannier` lowest bands must be separated from the
  higher ones.

## Background

The package was developed for the master's thesis *Ultracold Atoms in a Tunable Brickwall Lattice* (Yarin Cohen,
MPQ / TUM / LMU, 2026), which describes the methods and applications in detail.

## Acknowledgements

Developed as part of a master's thesis in the Theory Division of the Max Planck Institute of Quantum Optics,
under the supervision of J. Ignacio Cirac and with the mentorship of Dominik Wild and Bennet Windt.
Thanks to Pascal Weckesser and David Gröters for discussions on the experimental side.

## How to cite

If you use this package in your work, please cite it as:

> Y. Cohen, *optical_lattices*, version 1.0.0 (2026), https://github.com/yarinch/optical_lattices

```bibtex
@software{optical_lattices,
  author  = {Cohen, Yarin},
  title   = {optical\_lattices},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/yarinch/optical_lattices}
}
```

## Contact

Questions and bug reports: please open an issue on GitHub.

## License

MIT, see [LICENSE](LICENSE).
