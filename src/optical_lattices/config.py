from dataclasses import dataclass
from functools import cached_property

import numpy as np

A0_NM = 0.0529  # Bohr radius in nm


@dataclass(frozen=True)
class Config:
    """
    Parameters of the optical lattice and of the numerics. Every module takes a Config.

    Units: lengths in the lattice constant a = pi/k (a = 1 by default),
    energies in recoil units E_r = hbar^2 k^2 / (2m), hbar = 1 (m is chosen so that E_r = 1).

    Config is immutable; to change a value, make a copy with ``dataclasses.replace(config, V1=...)``.

    Attributes:
        V1 (float): Depth of the L1 lattice, in E_r.
        V2 (float): Depth of the L2 square lattice, in E_r. This is the depth of the resulting lattice,
            not of the generating beams.
        phi (float): Phase between L1 and L2. phi = pi gives the brick-wall lattice, phi = 0 the triangular one.
        k (float): Advanced; leave at the default pi (a = 1). Lattice wavevector pi/a, i.e. the Brillouin
            zone edge, NOT the laser wavevector. Other values only rescale lengths (a = pi/k); energies in E_r
            are unchanged.
        w_z (float): hbar * omega_z of the harmonic confinement along z, in E_r. Only enters the Hubbard U,
            which scales as sqrt(w_z); the default 1 gives U per sqrt(w_z), to be rescaled with the actual value.
        L_max (int): Plane-wave cutoff; momenta 2k * n with n = -L_max..L_max in each direction.
        num_q (int): Number of quasimomentum points per direction of the Brillouin zone (Wannier90 k-grid).
        num_wannier (int): Number of bands (and Wannier functions) to compute, e.g. 2 for the brick-wall lattice,
            1 for the triangular lattice. These lowest bands should be well separated from the higher bands.
        wannier90_executor (str): How to run Wannier90. "Path" finds wannier90.x in wannier90_dir, in the
            environment variable WANNIER90_X or on the PATH (any OS); "WSL" runs wannier90.x inside WSL (Windows).
        wannier90_dir (str | None): Directory containing the Wannier90 binary, for the "Path" executor.
        a_s_bohr (float): s-wave scattering length in Bohr radii (default: 95, for 87Rb).
        lattice_constant_nm (float): Lattice constant a in nm, used to convert a_s from Bohr radii to lattice
            units (default: 752, the brick-wall lattice from a 1064 nm laser).
    """
    ### Lattice parameters ###
    V1: float
    V2: float
    phi: float
    k: float = np.pi
    w_z: float = 1

    ### Numerical parameters ###
    L_max: int = 10
    num_q: int = 20
    num_wannier: int = 2

    ### System configuration ###
    wannier90_executor: str = "Path"
    wannier90_dir: str | None = None

    ### Atomic parameters (only for the Hubbard U) ###
    a_s_bohr: float = 95.0
    lattice_constant_nm: float = 752.0

    @cached_property
    def q_range(self) -> np.ndarray:
        """Plane-wave indices n = -L_max..L_max."""
        return np.arange(-self.L_max, self.L_max + 1)

    @cached_property
    def a(self) -> float:
        """Lattice constant pi/k (NOT the laser wavelength)."""
        return np.pi / self.k

    @cached_property
    def m(self) -> float:
        """Mass of the atom in these units (hbar = 1, E_r = k^2 / (2m) = 1)."""
        return self.k ** 2 / 2

    @cached_property
    def a_s(self) -> float:
        """Scattering length in the package's length units (the lattice constant is a = pi/k)."""
        return self.a_s_bohr * A0_NM / self.lattice_constant_nm * self.a
