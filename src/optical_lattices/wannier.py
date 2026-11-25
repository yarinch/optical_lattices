from typing import Iterable
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
import subprocess
import tempfile
import shutil
import os

import wannier90io
import numpy as np
from scipy.ndimage import shift

from optical_lattices.config import Config
from optical_lattices.energy import Energy


class WannierExecutor(metaclass=ABCMeta):
    """Runs the Wannier90 binary in a seed directory.

    Subclasses provide ``_wannier90_x``: the shell command that starts wannier90.x (a quoted path, or e.g.
    ``wsl -d ubuntu wannier90.x``). It is run through the shell with the seed directory as working directory.
    """

    def __init__(self, config: Config):
        self.config = config

    def preprocess(self, seed_dir: str) -> None:
        """Run the Wannier90 preprocessing step."""
        result = subprocess.run(f"{self._wannier90_x} -pp seed.win", cwd=seed_dir, shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"Error during preprocessing: {result.stderr.decode()}")
            self._try_print_output(seed_dir)
            raise RuntimeError("Wannier90 preprocessing failed.")

        if result.stdout:
            print(f"Unexpected output during preprocessing: {result.stdout.decode()}")
            self._try_print_output(seed_dir)
            raise RuntimeError("Wannier90 preprocessing produced unexpected output.")

    def execute(self, seed_dir: str) -> None:
        """Run the Wannier90 execution step."""
        result = subprocess.run(f"{self._wannier90_x} seed.win", cwd=seed_dir, shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"Error during execution: {result.stderr.decode()}")
            self._try_print_output(seed_dir)
            raise RuntimeError("Wannier90 failed.")

        if result.stdout:
            print(f"Unexpected output during execution: {result.stdout.decode()}")
            self._try_print_output(seed_dir)
            raise RuntimeError("Wannier90 produced unexpected output.")

    @staticmethod
    def _try_print_output(seed_dir: str) -> None:
        print("Content of seed.wout:")
        try:
            with open(os.path.join(seed_dir, 'seed.wout'), 'r') as f:
                print(f.read())
        except FileNotFoundError:
            pass

        print("Content of seed.werr:")
        try:
            with open(os.path.join(seed_dir, 'seed.werr'), 'r') as f:
                print(f.read())
        except FileNotFoundError:
            pass

    @property
    @abstractmethod
    def _wannier90_x(self) -> str:
        """Return the command to run wannier90.x, which may differ based on the execution environment."""
        ...


class PathExecutor(WannierExecutor):
    """Runs a local wannier90.x binary (any operating system).

    The binary is looked up in this order:
      1. ``config.wannier90_dir``, if it contains ``wannier90.x.exe`` or ``wannier90.x``;
      2. the environment variable ``WANNIER90_X`` (full path to the binary);
      3. ``wannier90.x`` on the ``PATH``.
    """

    BINARY_NAMES = ("wannier90.x.exe", "wannier90.x")

    @property
    def wannier90_path(self) -> str:
        """Full path of the wannier90.x binary that will be run."""
        if self.config.wannier90_dir:
            for name in self.BINARY_NAMES:
                candidate = os.path.join(self.config.wannier90_dir, name)
                if os.path.isfile(candidate):
                    return candidate

        env = os.environ.get("WANNIER90_X")
        if env:
            return env

        on_path = shutil.which("wannier90.x")
        if on_path:
            return on_path

        raise FileNotFoundError(
            "Could not find wannier90.x. Install Wannier90 (https://wannier.org) and either set "
            "Config.wannier90_dir to its directory, set the environment variable WANNIER90_X to the binary, "
            "or put wannier90.x on the PATH."
        )

    @property
    def _wannier90_x(self) -> str:
        return f'"{self.wannier90_path}"'


class WSLExecutor(WannierExecutor):
    """Runs ``wannier90.x`` inside the WSL distribution ``ubuntu`` (Windows)."""

    @property
    def _wannier90_x(self) -> str:
        return "wsl -d ubuntu wannier90.x"


@dataclass
class NNKP:
    k_points: np.ndarray  # k-point grid in Cartesian coordinates (shape: [num_kpoints, 2])
    nnkpts: list  # List of neighbor k-points and their reciprocal lattice vectors (shape: [num_kpoints, num_neighbors, 3])

    @classmethod
    def from_raw(cls, config: Config, raw: str) -> 'NNKP':
        nnkp = wannier90io.parse_nnkp_raw(raw)

        k_points = np.array(nnkp['kpoints']['kpoints']) * 2 * config.k  # Convert to actual coordinates
        if not np.all(k_points[:, 2] == 0):
            raise RuntimeError("k-points are not in the xy-plane as expected.")

        return cls(
            k_points=k_points[:, :2],  # Only keep x and y components
            nnkpts=nnkp['nnkpts']
        )


@dataclass
class WannierResult:
    """Result of :meth:`Wannier.wannierize`.

    Attributes:
        eigensystem: List of (eigenvalues, eigenvectors) from :meth:`Energy.diagonalize`, one per k-point
            (length num_kpoints), in the order of ``nnkp.k_points``.
        nnkp: Parsed .nnkp file content (k-point grid and neighbors).
        centers: Wannier centers in Cartesian coordinates, shape (num_wannier, 2). Wannier90 translates them
            into the home unit cell.
        unitaries: List of (k_point, U) for each k-point, U of shape (num_wannier, num_wannier) mixing the bands
            into Wannier functions.
        hamiltonian: The tight-binding Hamiltonian in the Wannier basis (Wannier90's seed_hr.dat):
            ``hamiltonian[(R, m, n)] = <w_m(0)| H |w_n(R)>`` in recoil energies, with R = (Rx, Ry) the lattice
            vector in units of a and m, n = 1..num_wannier (1-based, in the order of the initial guesses).
            The hopping is t = -hamiltonian[(R, m, n)]; the on-site terms (R = (0, 0), m = n) include the
            constant offset V1 + V2.
    """
    eigensystem: list
    nnkp: NNKP
    centers: np.ndarray
    unitaries: list
    hamiltonian: dict

    @property
    def overlaps(self) -> dict:
        """Deprecated alias of :attr:`hamiltonian`."""
        return self.hamiltonian

    @overlaps.setter
    def overlaps(self, value: dict) -> None:
        self.hamiltonian = value


class Wannier:
    def __init__(self, config: Config, initial_guesses: Iterable, executor: WannierExecutor = None):
        """
        Maximally localized Wannier functions of the lowest ``config.num_wannier`` bands, computed with Wannier90.
        :param config: Configuration object.
        :param initial_guesses: ``config.num_wannier`` (x, y) positions (in units of the lattice constant a) used
            as initial guesses for the Wannier centers, typically the potential minima. Their order sets the
            Wannier function index 1, 2, ... in the results.
        :param executor: A WannierExecutor that runs Wannier90. If None, chosen by ``config.wannier90_executor``.
        """
        if len(initial_guesses) != config.num_wannier:
            raise ValueError(f"Expected {config.num_wannier} initial guesses")

        self.config = config
        self.energy = Energy(self.config)
        self.initial_guesses = initial_guesses
        self._executor = executor if executor is not None else self._get_executor(config)

    def wannierize(self) -> WannierResult:
        """
        Run the full Wannier90 calculation in a temporary directory: write the input files, diagonalize on the
        k-point grid, run Wannier90 and parse its output.
        :return: WannierResult with the Wannier centers, the unitaries and the tight-binding Hamiltonian.
        """
        with tempfile.TemporaryDirectory() as seed_dir:
            with open(os.path.join(seed_dir, "seed.win"), 'w') as f:
                f.write(self._get_win())

            self._executor.preprocess(seed_dir)

            with open(os.path.join(seed_dir, 'seed.nnkp'), 'r') as nnkp_file:
                nnkp = NNKP.from_raw(self.config, nnkp_file.read())

            eigensystem = self._diagonalize(nnkp)
            self._generate_seed_files(seed_dir, nnkp, eigensystem)
            self._executor.execute(seed_dir)

            return WannierResult(
                eigensystem=eigensystem,
                nnkp=nnkp,
                centers=self._parse_xyz(seed_dir, self.config.num_wannier),
                unitaries=self._parse_u(seed_dir),
                hamiltonian=self._parse_wannier_overlaps(seed_dir)
            )

    def compute_wannier(self, wannier_result: WannierResult, x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        """Compute the Wannier functions on a given real-space grid.
        :param x, y: Arrays of the same shape (e.g. from np.meshgrid), in units of a.
        :return: List of num_wannier real arrays of that shape, normalized on the grid (assumes an
            ``indexing='xy'`` meshgrid). Slower than :meth:`compute_wannier_fast` but on any grid.
        """
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        wannier_functions = [np.zeros_like(x, dtype=np.complex128) for _ in range(self.config.num_wannier)]

        qx, qy = np.meshgrid(self.config.q_range, self.config.q_range, indexing='xy')
        qx = qx[:, :, np.newaxis, np.newaxis]  # shape: (num_q, num_q, 1, 1) for broadcasting
        qy = qy[:, :, np.newaxis, np.newaxis]  # shape: (num_q, num_q, 1, 1) for broadcasting
        phase_factor = np.exp(2j * self.config.k * (qx * x + qy * y)).reshape(-1, *x.shape)  # shape: (num_q^2, Ny, Nx)

        for k, U in wannier_result.unitaries:
            bloch_phases = np.exp(1j * (k[0] * x + k[1] * y))  # shape (Ny, Nx)
            k_index = np.where(np.all(wannier_result.nnkp.k_points == k, axis=1))[0][0]
            u_k = wannier_result.eigensystem[k_index][1]  # shape: (num_q^2, num_bands)

            for m in range(self.config.num_wannier):
                eigenstates = bloch_phases * np.tensordot(
                    u_k[:, m],
                    phase_factor,
                    axes=(0, 0)
                )

                for n in range(self.config.num_wannier):
                    wannier_functions[n] += U[m, n] * eigenstates

        if not all(np.allclose(w.imag, 0, atol=1e-6) for w in wannier_functions):
            raise RuntimeError("Wannier function has significant imaginary part")

        # Normalize
        normalized = []
        for w in wannier_functions:
            norm = np.sqrt(np.trapezoid(np.trapezoid(w.real ** 2, x[0, :]), y[:, 0]))
            normalized.append(w.real / norm)

        return normalized

    def get_plane_waves_coefficients(self, wannier_result: WannierResult) -> tuple[np.ndarray, np.ndarray]:
        """Return the plane-wave coefficients for each Wannier function, along the momentum grid axis."""
        N_G = len(self.config.q_range)  # Momentum cutoff
        master_dim = N_G * self.config.num_q  # All q-points considered for the IFFT

        # Initialize a list of master arrays, one for each well
        K_space_arrays = np.zeros((self.config.num_wannier, master_dim, master_dim), dtype=np.complex128)

        # Populate the master arrays
        for q_index, q in enumerate(wannier_result.nnkp.k_points):
            qx_idx = q_index % self.config.num_q
            qy_idx = q_index // self.config.num_q

            # Loop through every well
            for t in range(self.config.num_wannier):
                # Get the (N_G, N_G) plane-wave coefficients for well t
                product = (wannier_result.unitaries[q_index][1][:, t]  # U_[l, t](q)
                           @
                           wannier_result.eigensystem[q_index][1].T)  # c_n^[l, q]

                # Place into the corresponding master array, multiple copies for each G spacing
                K_space_arrays[t][qx_idx::self.config.num_q, qy_idx::self.config.num_q] = product.reshape(N_G, N_G).T

        # Normalization: there are (num_q)^2 elements - normalize by square root of it
        K_space_arrays /= self.config.num_q

        # Strictly monotonic 1D momentum axis
        G0 = 2 * self.config.k
        K_axis = G0 * np.linspace(-self.config.L_max, self.config.L_max + 1, num=master_dim, endpoint=False)

        return K_axis, K_space_arrays

    def compute_wannier_fast(self, wannier_result: WannierResult):
        """Compute the Wannier functions with an inverse FFT. Much faster than :meth:`compute_wannier`, but on a
        fixed grid: num_q unit cells per direction, centred on the origin, with spacing a / (2 L_max + 1).
        :return: (X, Y, wannier_functions): X, Y from ``np.meshgrid(x, y, indexing='ij')`` (first axis is x), and a
            list of num_wannier real, normalized arrays on that grid. The global phase is chosen to make them real.
        """
        N_G = len(self.config.q_range)  # Momentum cutoff
        master_dim = N_G * self.config.num_q  # All q-points considered for the IFFT
        _, K_space_arrays = self.get_plane_waves_coefficients(wannier_result)

        # IFFT to get real-space values
        K_zero_idx = self.config.L_max * self.config.num_q
        dx = self.config.a / N_G

        w_centered_list = []

        for t in range(self.config.num_wannier):
            # Roll the array to put K=(0,0) precisely at [0, 0]
            K_shifted = np.roll(K_space_arrays[t], shift=(-K_zero_idx, -K_zero_idx), axis=(0, 1))

            # Run the 2D IFFT
            w_real = np.fft.ifft2(K_shifted)

            # Rotate to make real
            peak = np.unravel_index(np.argmax(np.abs(w_real)), w_real.shape)
            w_real *= np.exp(-1j * np.angle(w_real[peak]))

            if not np.all(np.isclose(w_real.imag, 0, atol=1e-6)):
                raise ValueError(f"Wannier function {t} has a large imaginary part: {np.max(np.abs(w_real.imag))}")

            w_real = w_real.real

            # Normalize
            norm = np.sum(w_real ** 2) * dx ** 2
            w_normalized = w_real / np.sqrt(norm)

            # Shift the Wannier array so the origin (0,0) is in the middle
            w_centered = np.fft.fftshift(w_normalized)
            w_centered_list.append(w_centered)

        # Shared 1D spatial axes
        x = y = np.arange(-master_dim // 2, master_dim // 2) * dx
        X, Y = np.meshgrid(x, y, indexing='ij')

        return X, Y, w_centered_list

    def hubbard_U(self, wannier_result: WannierResult) -> np.ndarray:
        """Compute the on-site interaction energy U of each Wannier function, in recoil energies.

        U = g * sqrt(m w_z / (2 pi)) * integral |w(x, y)|^4 dx dy, with the contact pseudopotential
        g = 4 pi a_s / m and a harmonic ground state of frequency w_z along z. The result depends on the
        assumed ``config.a_s`` and ``config.w_z``.
        """
        X, Y, wannier_functions = self.compute_wannier_fast(wannier_result)
        dx = X[1, 0] - X[0, 0]
        g = 4 * np.pi * self.config.a_s / self.config.m  # pseudopotential strength
        z_confinement = np.sqrt(self.config.m * self.config.w_z / (2 * np.pi))
        Us = np.zeros(self.config.num_wannier, dtype=np.float64)
        for i, w in enumerate(wannier_functions):
            integral = np.sum(w ** 4) * (dx ** 2)
            Us[i] = g * z_confinement * integral

        return Us

    def _generate_seed_files(self, seed_dir: str, nnkp: NNKP, eigensystem: list) -> None:
        """Generate the necessary seed files for Wannier90."""
        with open(os.path.join(seed_dir, "seed.mmn"), 'w') as f:
            f.write(self._build_mmn(nnkp, eigensystem))

        with open(os.path.join(seed_dir, "seed.amn"), 'w') as f:
            f.write(self._build_amn(nnkp, eigensystem))

        with open(os.path.join(seed_dir, "seed.eig"), 'w') as f:
            f.write(self._build_eig(eigensystem))

    def _diagonalize(self, nnkp: NNKP) -> list:
        """Precompute all eigenvalues on the k-point grid."""
        return [self.energy.diagonalize(qx, qy, num_bands=self.config.num_wannier) for qx, qy in nnkp.k_points]

    def _get_neighbor_overlap(self, u1: np.ndarray, u2: np.ndarray, G: list) -> list:
        """Calculate the overlaps between two momentum eigenvalues with a reciprocal lattice vector G."""
        q_size = len(self.config.q_range)
        g_x, g_y, _ = G  # g_z is ignored  -- periodic boundary conditions with just one Z-layer

        overlaps = []
        for band2 in range(self.config.num_wannier):
            # shift u2 by the reciprocal lattice vector G and pad with zeros
            coefficients = u2[:, band2].reshape((q_size, q_size))
            shifted_coefficients = shift(coefficients, shift=(-g_y, -g_x), order=0, mode='constant', cval=0.0)

            for band1 in range(self.config.num_wannier):
                overlap = u1[:, band1].conj() @ shifted_coefficients.ravel()
                overlaps.append(overlap)

        return overlaps

    def _get_neighbor_overlap_string(self, eigensystem: list, line: list) -> str:
        k, neighbor_k, G = line[0], line[1], line[2:]
        overlaps = self._get_neighbor_overlap(eigensystem[k - 1][1], eigensystem[neighbor_k - 1][1], G)
        return f"{k}  {neighbor_k}  {'  '.join(str(g) for g in G)}\n" + "\n".join(
            f"{np.real(overlap):.12f} {np.imag(overlap):.12f}" for overlap in overlaps
        )

    def _build_mmn(self, nnkp: NNKP, eigensystem: list) -> str:
        return ("Overlap Matrix Elements\n" +
                f"{self.config.num_wannier} {len(nnkp.k_points)} {len(nnkp.nnkpts) // len(nnkp.k_points)}\n" +
                "\n".join(self._get_neighbor_overlap_string(eigensystem, line) for line in nnkp.nnkpts) +
                "\n")

    def _get_k_point_projections_string(self, k_index: int, nnkp: NNKP, eigensystem: list) -> str:
        """Get the projections of an eigenstate with a delta function at every well."""
        q = nnkp.k_points[k_index]
        u = eigensystem[k_index][1]  # Eigenvectors at this k-point
        x_waves, y_waves = np.meshgrid(self.config.q_range, self.config.q_range, indexing='xy')

        projections = ""
        for wannier_idx, well in enumerate(self.initial_guesses):
            wx, wy = well
            for band_idx in range(self.config.num_wannier):
                # Compute overlap of u_q[:, band_idx] with delta function at (wx, wy)
                bloch_phase = np.exp(-1j * (q[:2] @ well))

                phases = np.exp(-2j * self.config.k * (wx * x_waves + wy * y_waves)) * bloch_phase
                projection = u[:, band_idx].conj() @ phases.ravel()

                projections += f"{band_idx + 1}  {wannier_idx + 1}  {k_index + 1}  {np.real(projection):.12f} {np.imag(projection):.12f}\n"

        return projections

    def _build_amn(self, nnkp: NNKP, eigensystem: list) -> str:
        return ("Projection Matrix Elements\n" +
                f"{self.config.num_wannier} {len(nnkp.k_points)} {self.config.num_wannier}\n" +
                ''.join(
                    self._get_k_point_projections_string(k_index, nnkp, eigensystem)
                    for k_index in range(len(nnkp.k_points))))

    @staticmethod
    def _build_eig(eigensystem: list) -> str:
        return '\n'.join(
            f"{band_idx}  {k_point_number}  {energy:.12f}"
            for k_point_number, (eigenvalues, _) in enumerate(eigensystem, start=1)
            for band_idx, energy in enumerate(eigenvalues, start=1)
        )

    def _get_win(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, 'seed.win.template'), 'r') as f:
            template = f.read()

        # Convert to 3D fraction coordinates
        initial_guesses = [f"f = {x / self.config.a:.6f}, {y / self.config.a:.6f}, 0.000000: s"
                           for (x, y) in self.initial_guesses]

        return template.format(
            num_wannier=self.config.num_wannier,
            num_q=self.config.num_q,
            a=self.config.a,
            initial_guesses="\n  ".join(initial_guesses),
            kpoints='\n'.join(self._get_kpoints(self.config.num_q))
        )

    @staticmethod
    def _get_kpoints(num_q: int) -> list:
        """Create Monkhorst-Pack k-point grid in fractional coordinates."""
        kpoints = []
        for y in range(num_q):
            for x in range(num_q):
                # Standard Monkhors-Pack grid logic [0, 1)
                kx = x / num_q
                ky = y / num_q
                kpoints.append(f"{kx:.6f}  {ky:.6f}  0.000000")

        return kpoints

    @staticmethod
    def _parse_xyz(seed_dir: str, num_wannier: int) -> np.ndarray:
        """Parse the Wannier centers from the seed_centres.xyz file and return their (x, y) coordinates."""
        with open(os.path.join(seed_dir, 'seed_centres.xyz'), 'r') as f:
            data = f.readlines()[2:2 + num_wannier]

        points = np.array([[float(x) for x in line.split()[1:]] for line in data])

        if not np.all(points[:, 2] == 0):
            raise RuntimeError("Expected z=0 for all Wannier centers.")

        return points[:, :2]

    def _parse_u(self, seed_dir: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """Parse seed_u.mat file to get the unitary transformation matrices."""
        with open(os.path.join(seed_dir, 'seed_u.mat'), 'r') as f:
            data = f.readlines()[3:]  # Skip header lines

        block_size = (1  # Line with kpoint
                      + self.config.num_wannier ** 2  # Unitary
                      + 1  # Empty line
                      )
        return [self._parse_u_block(data[i:i + block_size]) for i in range(0, len(data), block_size)]

    def _parse_u_block(self, block: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse a single block of the seed_u.mat file corresponding to one k-point.
        :param block: 2 + num_wannier^2 lines of the seed_u.mat file.
        :return: Tuple of (k-point in Cartesian coordinates, 2x2 unitary matrix for that k-point).
        """
        k_point = np.array([float(val) for val in block[0].strip().split()]) * 2 * self.config.k
        if not np.all(k_point[2:] == 0):
            raise ValueError("Expected z=0 for all k-points.")
        k_point = k_point[:2]  # Only keep x and y components

        # Reshape with Fortran order since the file is written in column-major format
        u = np.array([self._parse_complex(line) for line in block[1:1 + self.config.num_wannier ** 2]]).reshape(
            (self.config.num_wannier, self.config.num_wannier), order='F')
        assert np.allclose(u.conj().T @ u, np.eye(self.config.num_wannier), atol=1e-6), "Parsed matrix is not unitary."

        return k_point, u

    @staticmethod
    def _parse_complex(line: str) -> complex:
        real, imag = line.strip().split()
        return float(real) + 1j * float(imag)

    @classmethod
    def _parse_wannier_overlaps(cls, seed_dir: str) -> dict[tuple[tuple[int, int], int, int], float]:
        with open(os.path.join(seed_dir, 'seed_hr.dat'), 'r') as f:
            data = f.readlines()

            # Skip headers and degeneracy list
            nrpts = int(data[2].strip())
            header_lines = 3 + int(np.ceil(nrpts / 15))
            data = data[header_lines:]

        return {(R, m, n): overlap for R, m, n, overlap in (cls._parse_overlap_line(line) for line in data)}

    @staticmethod
    def _parse_overlap_line(line: str) -> tuple[tuple[int, int], int, int, float]:
        """Return in format: (R, m, n, overlap)"""
        rx, ry, rz, m, n, real, imag = line.strip().split()
        R = (int(rx), int(ry))
        if int(rz) != 0:
            raise ValueError("Expected rz=0 for 2D system.")
        if not np.isclose(float(imag), 0, atol=1e-6):
            raise ValueError(f"Expected real overlap values. {imag=}, {real=}")
        return R, int(m), int(n), float(real)

    @staticmethod
    def _get_executor(config) -> WannierExecutor:
        match config.wannier90_executor:
            case "Path":
                return PathExecutor(config)
            case "WSL":
                return WSLExecutor(config)
            case _:
                raise ValueError(f"Unknown Wannier90 executor: {config.wannier90_executor}")
