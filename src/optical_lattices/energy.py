from functools import cached_property

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh

from optical_lattices.config import Config


class Energy:
    """Band structure of the lattice from a plane-wave expansion.

    At quasimomentum (qx, qy) the Hamiltonian is written in the plane-wave basis
    exp(i (q + 2k n) . r) with n = (n_x, n_y), n_x, n_y in ``config.q_range``
    (i.e. -L_max..L_max). A state is a vector of length (2 L_max + 1)^2, flattened
    with n_x running fastest: index = (n_y + L_max) * (2 L_max + 1) + (n_x + L_max).

    The potential matrix elements are the analytical Fourier components of
    :class:`~optical_lattices.potential.Potential`. Energies are in recoil units
    and include the same constant ``offset = V1 + V2``.
    """

    def __init__(self, config: Config):
        self.config = config

    @property
    def offset(self) -> float:
        """Constant offset V1 + V2 included in all energies (same as ``Potential.offset``)."""
        return self.config.V1 + self.config.V2

    def get_energies(self, qx: float, qy: float, num_bands: int = 1) -> np.ndarray:
        """Compute the eigenenergies at given quasimomentum (qx, qy).
        :param qx: Quasimomentum in x direction.
        :param qy: Quasimomentum in y direction.
        :param num_bands: Number of energy bands to compute.
        :return: The lowest ``num_bands`` eigenenergies in increasing order (including the offset).
        """
        energies, _ = self.diagonalize(qx, qy, num_bands)
        return energies

    def diagonalize(self, qx: float, qy: float, num_bands: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Compute the eigenenergies and eigenstates at given quasimomentum (qx, qy).
        :param qx: Quasimomentum in x direction.
        :param qy: Quasimomentum in y direction.
        :param num_bands: Number of energy bands to compute.
        :return: Tuple (energies, vectors) in increasing energy order. ``vectors[:, i]`` holds the
            plane-wave coefficients of band i, in the basis order described in the class docstring.
        """
        # Shift-invert around 0 returns the eigenvalues closest to 0. These are the lowest bands
        # only because the offset makes H >= 0; do not remove the offset.
        energies, vecs = eigsh(self._H(qx, qy), sigma=0, k=num_bands)
        # sort according to energies
        idx = energies.argsort()
        return energies[idx], vecs[:, idx]

    def _H(self, qx: float, qy: float) -> sp.csr_array:
        """Construct the Hamiltonian matrix for the system in momentum space."""
        return self._kinetic_energy(qx, qy) + self._potential_energy

    def _kinetic_energy(self, qx: float, qy: float) -> sp.csr_array:
        """Calculate kinetic energy part of the Hamiltonian."""
        k, q_range = self.config.k, self.config.q_range
        x_waves, y_waves = np.meshgrid(q_range, q_range, indexing='xy')
        energies = ((qx + 2 * k * x_waves) ** 2 + (qy + 2 * k * y_waves) ** 2) / (2 * self.config.m)
        return sp.diags(energies.ravel()).tocsr()

    @cached_property
    def _potential_energy(self) -> sp.csr_array:
        """Construct the potential energy matrix for the system in momentum space."""
        offset = self.offset * sp.eye(len(self.config.q_range) ** 2)  # Makes H >= 0
        return offset - self._overlap_L1() - self._overlap_L2()

    def _overlap_L1(self) -> sp.csr_array:
        """Plane-wave matrix elements of the L1 potential V1 cos^2(k(x+y) + phi/2) (analytical)."""
        n = len(self.config.q_range)
        off_diag = np.full(n - 1, 0.5)
        overlap_above = sp.diags([off_diag], [1])

        off_diag = np.full(n - 1, 0.5)
        overlap_below = sp.diags([off_diag], [-1])

        I = sp.eye(n)
        phase = np.exp(1j * self.config.phi)

        return self.config.V1 * (
                0.5 * sp.kron(I, I) + phase.conj() * sp.kron(overlap_above, overlap_above) + phase * sp.kron(overlap_below, overlap_below)
        )

    def _overlap_L2(self) -> sp.csr_array:
        """Plane-wave matrix elements of the L2 potential V2 cos^2(kx) cos^2(ky) (analytical)."""
        n = len(self.config.q_range)
        diag = np.full(n, 0.5)
        off_diag = np.full(n - 1, 0.25)
        overlap_y = overlap_x = sp.diags([off_diag, diag, off_diag], [-1, 0, 1])

        return self.config.V2 * sp.kron(overlap_y, overlap_x).tocsr()
