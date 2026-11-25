from dataclasses import dataclass
from functools import cached_property

from scipy.interpolate import RectBivariateSpline
import scipy.linalg
import numpy as np

from optical_lattices import Wannier, WannierResult


@dataclass
class MoireSite:
    layer: int  # 1 or 2 - layer index
    sublattice: str  # A or B sublattice
    j: int  # index within unit cell within specific layer/sublattice combo
    site: np.ndarray  # (x, y) index


class TwistedBilayer:
    """Tight-binding model of two brick-wall layers twisted against each other.

    Layer 2 is layer 1 rotated by the commensurate angle theta with cos(theta) = 2mn / (m^2 + n^2) about the point
    ``delta``. The moiré unit cell is spanned by (n, m) and (m, -n) (units of a). Within each layer, the hoppings are
    the Wannier Hamiltonian ``wr.hamiltonian`` of the single layer; between the layers, the coupling of two sites is
    Omega * <w_i | w_j>, the spatial overlap of their Wannier functions (layer 2 rotated), with Omega an energy (E_r).

    Requirements on the input:
        - ``wr`` must have exactly two Wannier functions, from the initial guesses ((-p, -p), (p, p)) in this order.
        - The lattice of ``w.config`` must have double wells (V1/V2 >= 1/2).
    """

    def __init__(self, w: Wannier, wr: WannierResult, n: int, m: int, delta=np.array([0, 0]), neighbor_cells: int = 1):
        """
        :param w: The Wannier object of the single layer.
        :param wr: Its WannierResult (see the requirements above).
        :param n, m: Integers that set the commensurate twist angle and the moiré cell.
        :param delta: Rotation center (x, y).
        :param neighbor_cells: Number of moiré cells in each direction, for which couplings are included,
            when computing the tight-binding Hamiltonian. Must be large enough to include all significant couplings.
            Default is 1, corresponding to the 3 x 3 moiré cells around the central one.
        """
        self.w = w
        self.wr = wr
        self.config = w.config
        self.n = n
        self.m = m
        self.delta = delta

        # Rotation matrix
        theta = np.arccos(2 * m * n / (m ** 2 + n ** 2))
        self.M = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        self.theta = theta

        self.D = self.M @ self.delta - self.delta

        # Moire cell vectors, in units of a
        self.V1 = np.array((n, m))
        self.V2 = np.array((m, -n))
        self.sites = self._get_all_unit_cell_sites()

        # Reciprocal lattice vectors
        reciprocal_prefactor = 2 * self.config.k / (n**2 + m**2)
        self.G1 = reciprocal_prefactor * self.V1
        self.G2 = reciprocal_prefactor * self.V2

        # Setup Wannier coefficients once
        self.K_axis, (self.w1, self.w2) = w.get_plane_waves_coefficients(self.wr)
        self.c1_rot = self._get_rotated_coefficients(self.w1)
        self.c2_rot = self._get_rotated_coefficients(self.w2)

        # Moiré cells (in units of V1, V2) whose sites are coupled to the central cell
        self.delta_n = [np.array((nx, ny))
                        for nx in range(-neighbor_cells, neighbor_cells + 1)
                        for ny in range(-neighbor_cells, neighbor_cells + 1)]

    def transform(self, point: np.ndarray) -> np.ndarray:
        """Transform a point from layer 1 to layer 2 coordinates."""
        # Multiply by transpose from the right to support array of points
        return point @ self.M.T - self.D

    def inv_transform(self, point: np.ndarray) -> np.ndarray:
        """Transform a point from layer 2 to layer 1 coordinates."""
        return self.M.T @ (point + self.D)

    def _get_in_cell(self, point, rotate=False):
        """Get subset of points within the Moire unit cell using fractional coordinates."""
        max_dist = 2 * (self.n + self.m)
        Rs = np.arange(-max_dist, max_dist + 1) * self.config.a
        Rs = np.array(np.meshgrid(Rs, Rs, indexing='ij')).T.reshape(-1, 2)

        # Generate base physical coordinates for layer 1
        points = np.array([point + R for R in Rs])

        # Apply spatial transformation for layer 2
        if rotate:
            points = self.transform(points)

        # Define the Moire basis matrix W (columns are the Moire lattice vectors)
        W = self.config.a * np.array([self.V1, self.V2]).T
        W_inv = np.linalg.inv(W)

        # Convert physical points to fractional coordinates relative to the shifted origin (delta)
        # We use W_inv.T because points is an (N, 2) array of row vectors
        frac_coords = (points - self.delta) @ W_inv.T

        # Enforce periodic boundary conditions: 0 <= alpha < 1 and 0 <= beta < 1
        # Epsilon absorbs floating point noise for points sitting exactly on the boundary
        eps = 1e-10
        in_cell = (
                (frac_coords[:, 0] >= -eps) & (frac_coords[:, 0] < 1 - eps) &
                (frac_coords[:, 1] >= -eps) & (frac_coords[:, 1] < 1 - eps)
        )

        return points[in_cell]

    def _get_all_unit_cell_sites(self) -> list[MoireSite]:
        """Assuming rotation around origin between sites"""
        r = self.config.V1 / self.config.V2
        p = (1 / self.config.k) * np.arccos(np.sqrt((2 * r) / (4 * r - 1)))
        A_site = np.array([-p, -p])
        B_site = np.array([p, p])

        all_A_sites = self._get_in_cell(A_site)
        all_B_sites = self._get_in_cell(B_site)
        all_MA_sites = self._get_in_cell(A_site, rotate=True)
        all_MB_sites = self._get_in_cell(B_site, rotate=True)

        assert len(all_MA_sites) == len(all_MB_sites) == len(all_A_sites) == len(all_B_sites), "Mismatching unit cell site amounts"
        Ns = len(all_A_sites)
        indices = [(layer, s, j) for layer in [1, 2] for s in ['A', 'B'] for j in range(Ns)]
        sites = np.concat([all_A_sites, all_B_sites, all_MA_sites, all_MB_sites])
        return [MoireSite(layer, s, j, site) for (layer, s, j), site in zip(indices, sites)]

    def _get_site_R(self, site: MoireSite) -> np.ndarray:
        """Get unit cell displacement, only in original frame (rotate back if necessary), in units of config.a."""
        coords = site.site if site.layer == 1 else self.inv_transform(site.site)
        return np.round(coords / self.config.a).astype(int)

    def _get_rotated_coefficients(self, c) -> np.ndarray:
        # We must interpolate to get the rotated axes for layer 2
        spline_real = RectBivariateSpline(self.K_axis, self.K_axis, c.real)
        spline_imag = RectBivariateSpline(self.K_axis, self.K_axis, c.imag)

        # 2. Generate the Rotated Coordinates for layer 2
        K_grid = np.stack(np.meshgrid(self.K_axis, self.K_axis, indexing='ij'))  # Shape: (2, N, N)
        Kx_rot, Ky_rot = np.tensordot(self.M.T, K_grid, axes=1)  # Shape: (2, N, N)

        # 3. Evaluate the rotated coefficients
        c_rot_real = spline_real.ev(Kx_rot, Ky_rot)
        c_rot_imag = spline_imag.ev(Kx_rot, Ky_rot)
        c_rot = c_rot_real + 1j * c_rot_imag

        return c_rot

    @cached_property
    def _wij(self):
        """Since this is used often, precompute the overlaps between Wannier functions of the original and rotated frame"""
        l = len(self.w1)
        wij = np.zeros((2, 2, l, l), dtype=np.complex128)
        for i, wi in enumerate([self.w1, self.w2]):
            for j, wj in enumerate([self.c1_rot, self.c2_rot]):
                wij[i, j, :, :] = wi.conj() * wj

        return wij


    def _get_interlayer_coupling(self, i: MoireSite, j: MoireSite) -> np.complex128:
        """Note: this doesn't include Omega, which should be multiplied in to get the actual coupling"""
        if i.layer == j.layer:
            raise ValueError("Both sites are in the same layer")

        # The following assumes i is in layer 1 and j is in layer 2, otherwise conjugate.
        if i.layer == 2:
            return self._get_interlayer_coupling(j, i).conj()

        # Start by collecting all plane wave coefficients
        wi = 0 if i.sublattice == 'A' else 1
        wj = 0 if j.sublattice == 'A' else 1
        wij = self._wij[wi, wj]

        # Integrate with the Phase Shift
        R1 = self._get_site_R(i)
        R2 = self._get_site_R(j)
        dR = self.config.a * (R1 - self.M @ R2) + self.D

        # Factorized 1D phase calculation (avoids 2D meshgrid & 2D exponentials)
        # self.K_axis is 1D array of length N
        phase_x = np.exp(1j * (dR[0] * self.K_axis))  # Shape (N,)
        phase_y = np.exp(1j * (dR[1] * self.K_axis))  # Shape (N,)

        # Fused 2D Contraction with Einstein Summation
        return np.einsum('mn,m,n->', wij, phase_x, phase_y, optimize=True)

    def get_coupling(self, i: MoireSite, j: MoireSite):
        """Get element (i, j) in the Moire unit cell Hamiltonian"""
        if i.layer == j.layer:  # Same layer - get (unrotated) single-layer hopping
            dR = tuple(self._get_site_R(j) - self._get_site_R(i))
            si = 1 if i.sublattice == 'A' else 2
            sj = 1 if j.sublattice == 'A' else 2
            # Hoppings beyond the range of Wannier90's hr.dat are negligible
            return self.wr.hamiltonian.get((dR, si, sj), 0.0)
        else:
            return self._get_interlayer_coupling(i, j)


    @cached_property
    def _tensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute both D and J tensors in a single loop, in addition to the interlayer mask - to include Omega"""
        N = len(self.sites)
        J = np.zeros((N, N, len(self.delta_n)), dtype=np.complex128)  # All tunneling coefficients needed
        L = np.zeros((N, N, len(self.delta_n), 2), dtype=np.float64)  # All site distances
        interlayer_mask = np.zeros((N, N, 1), dtype=bool)

        for i, si in enumerate(self.sites):
            for j, sj in enumerate(self.sites):
                interlayer_mask[i, j, 0] = (si.layer != sj.layer)

                for dn_index, dn in enumerate(self.delta_n):
                    shift = dn[0] * self.V1 + dn[1] * self.V2
                    new_site = MoireSite(layer=sj.layer, sublattice=sj.sublattice, j=sj.j,
                                         site=sj.site + self.config.a * shift)

                    J[i, j, dn_index] = self.get_coupling(si, new_site)
                    L[i, j, dn_index, :] = (new_site.site - si.site)

        return J, L, interlayer_mask

    def J(self, Omega: np.float64) -> np.ndarray:
        """Get the tunneling matrix for a given Omega"""
        J, _, interlayer_mask = self._tensors
        return np.where(interlayer_mask, Omega * J, J)

    @property
    def L(self) -> np.ndarray:
        _, L, _ = self._tensors
        return L

    def H_k(self, k: np.ndarray, Omega: np.float64) -> np.ndarray:
        """Get Bloch Hamiltonian for momentum k"""
        J = self.J(Omega)
        H = np.sum(J * np.exp(1j * (self.L @ k)), axis=2)

        assert scipy.linalg.ishermitian(H, atol=1e-6)
        return H

    def compute_dos(self, Omega: np.float64, nk: int = 100, num_bins: int = 200, e_range: tuple = None):
        """
        Computes the Density of States (DOS) for the Moiré superlattice.

        Parameters:
            Omega (float): Interlayer coupling strength.
            nk (int): Resolution of the k-grid spanning the first Brillouin zone.
            num_bins (int): Resolution of the energy axis for the histogram.
            e_range (tuple): Optional (min, max) energy bounds for the histogram.

        Returns:
            bin_centers (np.ndarray): 1D array of energy bin centers.
            total_dos (np.ndarray): 1D array containing the total DOS.
            band_dos (np.ndarray): 2D array of shape (num_bins, N) containing the DOS per band.
        """
        # 1. Create a grid in fractional coordinates spanning the first BZ [-0.5, 0.5]
        frac_coords = np.linspace(-0.5, 0.5, nk)
        g1_grid, g2_grid = np.meshgrid(frac_coords, frac_coords, indexing='ij')

        N = len(self.sites)
        eigenvalues = np.zeros((nk, nk, N))

        # 2. Evaluate the Hamiltonian over the true rotated BZ
        for i in range(nk):
            for j in range(nk):
                k_vec = g1_grid[i, j] * self.G1 + g2_grid[i, j] * self.G2
                H = self.H_k(k_vec, Omega=Omega)
                eigenvalues[i, j, :] = np.sort(scipy.linalg.eigh(H)[0])

        # 3. Determine the global energy bounds if not provided
        flat_evals = eigenvalues.flatten()
        if e_range is None:
            e_range = (flat_evals.min(), flat_evals.max())

        # 4. Compute the Total DOS
        total_dos, bin_edges = np.histogram(flat_evals, bins=num_bins, range=e_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 5. Compute the Band-Resolved DOS
        band_dos = np.zeros((num_bins, N))
        for band_idx in range(N):
            band_dos[:, band_idx], _ = np.histogram(eigenvalues[:, :, band_idx].flatten(), bins=bin_edges)

        return bin_centers, total_dos, band_dos
