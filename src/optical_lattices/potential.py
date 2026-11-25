from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optical_lattices.config import Config


class Potential:
    """Real-space potential of the tunable optical lattice.

    The potential is

        V(x, y) = -V1 cos^2(k(x + y) + phi/2) - V2 cos^2(kx) cos^2(ky) + offset,

    with the constant ``offset = V1 + V2`` added so that V >= 0. Energies are in
    recoil units and lengths in units of the lattice constant a = pi/k (a = 1 by
    default).

    The band structure in :class:`~optical_lattices.energy.Energy` uses the same
    offset, so subtract ``offset`` from both to get physical energies.

    Note: :class:`Energy` builds its Hamiltonian from the analytical Fourier
    components of this potential and does not call this class; it is meant for
    evaluation and plotting in real space.
    """

    def __init__(self, config: Config):
        self.config = config

    def potential(self, x: ArrayLike, y: ArrayLike) -> NDArray:
        """Evaluate the potential at positions (x, y).

        ``x`` and ``y`` may be scalars or arrays of broadcastable shapes.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self._L1_potential(x, y) + self._L2_potential(x, y) + self.offset

    @cached_property
    def offset(self) -> float:
        """Constant offset V1 + V2 added to the potential (and to the band energies).

        This is an upper bound of the lattice depth; depending on phi, the actual
        depth can be smaller.
        """
        return self.config.V1 + self.config.V2

    def _L1_potential(self, x: NDArray, y: NDArray) -> NDArray:
        return -self.config.V1 * (np.cos(self.config.k * (x + y) + self.config.phi / 2) ** 2)

    def _L2_potential(self, x: NDArray, y: NDArray) -> NDArray:
        return -self.config.V2 * (np.cos(self.config.k * x) * np.cos(self.config.k * y)) ** 2
