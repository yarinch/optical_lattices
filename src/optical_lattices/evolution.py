"""Time evolution of states in the lattice under a time-dependent Config (e.g. a ramp of the lattice depths).

    psi0   = Wavefunction.from_wannier(single_well_result, [1])
    target = Wavefunction.from_wannier(double_well_result, np.array([1, 1]) / np.sqrt(2))
    result = Evolution(LinearRamp(start, end, duration=20.0)).run(psi0, dt=0.05)
    result.final.fidelity(target)

States are stored k-resolved: at each quasimomentum k of the Wannier90 grid, the plane-wave coefficients
(in the basis of :class:`~optical_lattices.energy.Energy`). The Hamiltonian is diagonal in k, so each k evolves
independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Callable, Iterable

import numpy as np

from optical_lattices.config import Config
from optical_lattices.wannier import WannierResult


@dataclass(frozen=True)
class Wavefunction:
    """A state of the lattice, stored k-resolved.

    Attributes:
        coefficients: Plane-wave coefficients at each k-point, shape (num_k, (2 L_max + 1)^2), normalized to 1
            over all k.
        k_points: The k-points (Cartesian, units of 1/a), shape (num_k, 2), in the order of ``coefficients``.
        L_max: Plane-wave cutoff of the coefficients.
    """
    coefficients: np.ndarray
    k_points: np.ndarray
    L_max: int

    @classmethod
    def from_wannier(cls, result: WannierResult, amplitudes: Iterable[complex]) -> "Wavefunction":
        """The superposition sum_n amplitudes[n] w_n of the Wannier functions of a WannierResult (normalized).

        The amplitudes are in the order of the Wannier functions (i.e. of the initial guesses).
        """
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        num_wannier = result.unitaries[0][1].shape[0]
        if amplitudes.shape != (num_wannier,):
            raise ValueError(f"Expected {num_wannier} amplitudes, got shape {amplitudes.shape}")

        coefficients = []
        for (k, U), k_eig, (_, vectors) in zip(result.unitaries, result.nnkp.k_points, result.eigensystem):
            if not np.allclose(k, k_eig):
                raise RuntimeError("Wannier90 unitaries and eigenvectors are not on the same k-point order.")
            # Wannier function n at this k: sum_m U[m, n] u_m(k); combine them with the amplitudes.
            coefficients.append(vectors @ (U @ amplitudes))

        coefficients = np.stack(coefficients, axis=0)
        L_max = (int(round(np.sqrt(coefficients.shape[1]))) - 1) // 2
        k_points = np.array([k for k, _ in result.unitaries])
        return cls(coefficients=cls._normalized(coefficients), k_points=k_points, L_max=L_max)

    def fidelity(self, other: "Wavefunction") -> float:
        """|<other|self>|^2 (both normalized)."""
        self._check_compatible(other)
        return float(np.abs(np.vdot(self._normalized(other.coefficients), self._normalized(self.coefficients))) ** 2)

    def real_space(self, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The wavefunction in real space, via an inverse FFT (as :meth:`Wannier.compute_wannier_fast`).

        The grid covers the num_q x num_q unit cells of the k-grid (periodic), centred on the origin, with spacing
        a / (2 L_max + 1).
        :param config: The lattice of the state (for its length scale); its L_max and num_q must match the state.
        :return: (X, Y, psi): X, Y from ``np.meshgrid(x, y, indexing='ij')`` (first axis is x), and the complex
            wavefunction on that grid, normalized to 1 over it.
        """
        num_k, N = self.coefficients.shape
        num_q = int(round(np.sqrt(num_k)))
        N_G = 2 * self.L_max + 1
        if config.L_max != self.L_max or config.num_q != num_q or num_q ** 2 != num_k or N_G ** 2 != N:
            raise ValueError("The config's L_max and num_q must match the wavefunction.")

        # Place the coefficients of each k-point at the momenta k + G on one grid with spacing 2k / num_q
        master_dim = N_G * num_q
        K_space = np.zeros((master_dim, master_dim), dtype=np.complex128)
        k_indices = np.rint(self.k_points / (2 * config.k) * num_q).astype(int) % num_q
        for (ix, iy), c in zip(k_indices, self.coefficients):
            K_space[ix::num_q, iy::num_q] = c.reshape(N_G, N_G).T  # coefficients are stored with n_x fastest

        # K = 0 at index [0, 0], inverse FFT, origin in the middle
        K_space = np.roll(K_space, shift=(-self.L_max * num_q, -self.L_max * num_q), axis=(0, 1))
        psi = np.fft.fftshift(np.fft.ifft2(K_space))

        dx = config.a / N_G
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx ** 2)
        x = np.arange(-master_dim // 2, master_dim // 2) * dx
        X, Y = np.meshgrid(x, x, indexing="ij")
        return X, Y, psi

    @staticmethod
    def _normalized(coefficients: np.ndarray) -> np.ndarray:
        return coefficients / np.sqrt(np.real(np.vdot(coefficients, coefficients)))

    def _check_compatible(self, other: "Wavefunction") -> None:
        if self.L_max != other.L_max or self.k_points.shape != other.k_points.shape \
                or not np.allclose(self.k_points, other.k_points):
            raise ValueError("The wavefunctions live on different k-grids or plane-wave cutoffs.")


class Ramp(ABC):
    """A time-dependent lattice from t = 0 to ``duration`` (hbar / E_r).

    Subclasses implement :meth:`config_at`; k and L_max must stay fixed. See :class:`LinearRamp`.
    """

    def __init__(self, duration: float):
        self.duration = float(duration)

    @abstractmethod
    def config_at(self, t: float) -> Config:
        """The lattice at time t."""

    @property
    def start(self) -> Config:
        """The lattice at t = 0."""
        return self.config_at(0.0)

    @property
    def end(self) -> Config:
        """The lattice at t = duration."""
        return self.config_at(self.duration)


class LinearRamp(Ramp):
    """Linear ramp of V1, V2 and phi from ``start`` to ``end`` over ``duration`` (hbar / E_r).

    All other parameters are taken from ``start``; k and L_max must agree.
    """

    def __init__(self, start: Config, end: Config, duration: float):
        if start.k != end.k or start.L_max != end.L_max:
            raise ValueError("start and end must have the same k and L_max.")
        super().__init__(duration)
        self._start = start
        self._end = end

    def config_at(self, t: float) -> Config:
        """The lattice at time t (before 0: start, after the duration: end)."""
        s = 1.0 if self.duration <= 0 else min(max(float(t) / float(self.duration), 0.0), 1.0)
        return replace(
            self._start,
            V1=float(self._start.V1 + s * (self._end.V1 - self._start.V1)),
            V2=float(self._start.V2 + s * (self._end.V2 - self._start.V2)),
            phi=float(self._start.phi + s * (self._end.phi - self._start.phi)),
        )


@dataclass(frozen=True)
class EvolutionResult:
    """Outputs of an evolution run.

    Attributes:
        times: Times of the stored snapshots (hbar / E_r).
        psi_k_history: Stored k-resolved coefficient arrays, each of shape (num_k, (2 L_max + 1)^2).
        k_points, L_max: The k-grid and cutoff of the states (set by :meth:`Evolution.run`).
    """

    times: np.ndarray
    psi_k_history: list[np.ndarray]
    k_points: np.ndarray | None = None
    L_max: int | None = None

    def wavefunction(self, index: int) -> Wavefunction:
        """The stored state with the given index as a Wavefunction."""
        if self.k_points is None:
            raise ValueError("k-grid unknown; use Evolution.run to get Wavefunctions.")
        return Wavefunction(coefficients=self.psi_k_history[index], k_points=self.k_points, L_max=self.L_max)

    @property
    def final(self) -> Wavefunction:
        """The state at the end of the evolution."""
        return self.wavefunction(-1)

    def fidelity(self, target: Wavefunction) -> np.ndarray:
        """|<target|psi(t)>|^2 for every stored time."""
        return np.array([self.wavefunction(i).fidelity(target) for i in range(len(self.psi_k_history))])


class Evolution:
    """Time evolution during a :class:`Ramp`, from t = 0 to ``ramp.duration``.

    The Hamiltonian H(t) = T + V(t) is integrated with a second-order split-operator scheme: kinetic
    half-steps in the plane-wave basis and the potential step on the dual real-space grid via FFT. k and L_max
    must stay fixed during the ramp.
    """

    def __init__(self, ramp: Ramp):
        self.ramp = ramp

    def run(self, psi0: Wavefunction, dt: float, store_every: int | None = None) -> EvolutionResult:
        """Evolve psi0 from t = 0 to ``ramp.duration``.
        :param psi0: Initial state.
        :param dt: Maximal time step; the actual step is duration / ceil(duration / dt).
        :param store_every: Store every n-th step; default: only the initial and final states.
        """
        if self.ramp.start.L_max != psi0.L_max:
            raise ValueError("The ramp's L_max differs from the state's.")
        duration = self.ramp.duration
        num_steps = max(int(np.ceil(duration / dt)), 1)
        result = evolve_time_dependent_config_strang_fft(
            psi_k0=psi0.coefficients,
            k_points=psi0.k_points,
            config_of_t=self.ramp.config_at,
            dt=duration / num_steps,
            num_steps=num_steps,
            store_every=store_every if store_every is not None else num_steps,
        )
        return EvolutionResult(times=result.times, psi_k_history=result.psi_k_history,
                               k_points=psi0.k_points, L_max=psi0.L_max)


def evolve_time_dependent_config_strang_fft(
    *,
    psi_k0: np.ndarray,
    k_points: np.ndarray | Iterable[tuple[float, float]],
    config_of_t: Callable[[float], Config],
    dt: float,
    num_steps: int,
    store_every: int = 1,
    progress_every: int = 0,
) -> EvolutionResult:
    """Time-dependent Strang splitting using FFTs for the potential step.

    ``k_points`` are the (kx, ky) of the rows of ``psi_k0``, e.g. an array of shape (num_k, 2).

    Key idea
    --------
    In this project the Hamiltonian has the plane-wave structure

        H_k(t) = T_k + V(t)

    where T_k is diagonal in the (qx,qy) plane-wave basis, while V(t) is generated
    by a real-space lattice potential. Applying exp(-i V dt) in the plane-wave
    basis is dense; instead, we switch to a real-space grid via FFT:

        psi_G --FFT--> psi_x
        psi_x <- exp(-i V(x,t) dt) * psi_x
        psi_x --IFFT--> psi_G

    This avoids forming any N×N dense matrix exponentials and typically speeds
    ramp evolutions by orders of magnitude for notebook-sized systems.

    Notes / assumptions
    -------------------
    - This matches the *analytical* potential overlaps used by :class:`~optical_lattices.energy.Energy`.
      We implement the potential step using FFTs with the same sparse coupling
      structure as Energy._overlap_L1/_overlap_L2.
    - Config quantities q_range, k, m are assumed time-independent (same as the
      optimized Strang path).
    - This is a Strang split (2nd-order) method, so dt must still be chosen for
      accuracy for time-dependent protocols.
    """

    if store_every < 1:
        raise ValueError("store_every must be >= 1")
    if num_steps < 0:
        raise ValueError("num_steps must be >= 0")

    psi_k = np.array(psi_k0, dtype=np.complex128, copy=True)
    k_points = list(k_points)
    Nk, N = psi_k.shape
    if len(k_points) != Nk:
        raise ValueError(f"k_points length {len(k_points)} does not match Nk={Nk}")

    cfg0 = config_of_t(0.0)

    NG = int(cfg0.q_range.size)
    if NG * NG != N:
        raise ValueError(
            f"psi_k0 implies N={N} plane waves, but config_of_t(0).q_range gives NG={NG} => NG^2={NG*NG}."
        )

    # Precompute x_waves/y_waves index grids matching Energy._kinetic_energy.
    x_waves, y_waves = np.meshgrid(cfg0.q_range, cfg0.q_range, indexing="xy")

    # Kinetic half-step phases per k (constant across time if k,m,q_range fixed).
    phase_half = np.empty((Nk, N), dtype=np.complex128)
    for i, (kx, ky) in enumerate(k_points):
        Tk = ((kx + 2 * cfg0.k * x_waves) ** 2 + (ky + 2 * cfg0.k * y_waves) ** 2) / (2 * cfg0.m)
        phase_half[i, :] = np.exp((-1j * dt * 0.5) * Tk.ravel())

    # ------------------------------------------------------------------
    # FFT-based potential step using the analytical overlap structure.
    #
    # Energy._potential_energy = offset - overlap_L1 - overlap_L2
    # overlap_L1 = V1 * (0.5*I + phase*K(+,+) + phase*conj*K(-,-))
    # overlap_L2 = V2 * kron(overlap_y, overlap_x) with tri-diagonal weights.
    #
    # These are *convolution-like* in the plane-wave grid, so can be applied as
    # a diagonal multiplication in the FFT-dual grid.
    #
    # We construct the effective real-space potential sample values by taking
    # the 2D inverse FFT of the coupling kernels. This yields a potential field
    # Veff(x) such that applying exp(-i Veff dt) in real space matches the
    # overlap action in G space.
    # ------------------------------------------------------------------

    # Build the (normalized) convolution kernels in G-space.
    # L2 kernel corresponds to weights: diag=0.5, offdiag=0.25 along each axis.
    ker1d = np.zeros(NG, dtype=np.complex128)
    ker1d[0] = 0.5
    ker1d[1 % NG] = 0.25
    ker1d[-1 % NG] = 0.25
    ker_L2 = np.outer(ker1d, ker1d)

    # L1 has contributions at (0,0) and (+1,+1) and (-1,-1) with phase factors.
    ker_L1_base = np.zeros((NG, NG), dtype=np.complex128)
    ker_L1_base[0, 0] = 0.5
    ker_L1_p = np.zeros((NG, NG), dtype=np.complex128)
    ker_L1_p[1 % NG, 1 % NG] = 0.25
    ker_L1_m = np.zeros((NG, NG), dtype=np.complex128)
    ker_L1_m[-1 % NG, -1 % NG] = 0.25

    # Precompute FFTs of time-independent kernels.
    # In x-space, convolution is multiplication by FFT(kernel).
    F_L2 = np.fft.fft2(ker_L2)
    F_L1_base = np.fft.fft2(ker_L1_base)
    F_L1_p = np.fft.fft2(ker_L1_p)
    F_L1_m = np.fft.fft2(ker_L1_m)

    # Workspace arrays in FFT (x) space.
    psi_G = psi_k.reshape(Nk, NG, NG)
    psi_x = np.empty_like(psi_G)

    def _UV_phase(cfg) -> np.ndarray:
        """Return exp(-i Veff dt) on the FFT grid (NG,NG) for the midpoint config."""

        # Potential operator in G-space: offset*I - V1*(...) - V2*(...) .
        # In x-space (FFT-dual), that becomes multiplication by:
        #   offset - V1*(0.5 + e^{i phi} e^{+i(x+y)} + e^{-i phi} e^{-i(x+y)}) - V2*L2(x)
        # but rather than derive explicit x, we use the FFT of the kernels.
        #
        # Effective eigenvalues on the FFT grid (NG,NG):
        offset = cfg.V1 + cfg.V2
        lam = (
            offset
            - cfg.V1 * (F_L1_base + np.exp(-1j * cfg.phi) * F_L1_p + np.exp(1j * cfg.phi) * F_L1_m)
            - cfg.V2 * F_L2
        )
        # lam should be real (Hermitian); keep small imag noise from FFT.
        lam = np.real(lam)
        return np.exp((-1j * dt) * lam)

    times: list[float] = []
    hist: list[np.ndarray] = []

    for step in range(num_steps + 1):
        t = step * dt
        if (step % store_every) == 0:
            times.append(t)
            hist.append(psi_k.copy())
        if step == num_steps:
            break

        # Kinetic half-step (vectorized)
        psi_k *= phase_half

        # Midpoint potential sample for 2nd-order accuracy
        cfg_mid = config_of_t(t + 0.5 * dt)
        UV_phase = _UV_phase(cfg_mid)

        # Potential step: FFT -> multiply -> IFFT
        psi_x[:, :, :] = np.fft.ifft2(psi_G, axes=(1, 2))
        psi_x[:, :, :] *= UV_phase[None, :, :]
        psi_G[:, :, :] = np.fft.fft2(psi_x, axes=(1, 2))

        # Kinetic half-step
        psi_k *= phase_half

        if progress_every and (step % progress_every == 0):
            print(f"evolved step {step+1}/{num_steps}")

    return EvolutionResult(times=np.asarray(times, dtype=float), psi_k_history=hist)
