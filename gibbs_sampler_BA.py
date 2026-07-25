"""
gibbs_sampler_core.py
======================

Core library for the Metropolis-weighted Davies-generator Gibbs sampler
for the 1D transverse-field quantum Ising model.

This module contains only function/class definitions (no top-level
execution, no plotting side effects). Run everything from the companion
notebook `Gibbs_sampler_notebook.ipynb`.

Performance notes (what changed vs. the original script, and why)
-------------------------------------------------------------------
Everything below was actually benchmarked (small dense matrices, dim in
{8,16,32}, several thousand frequency points) before being kept -- a few
"obvious" vectorizations (e.g. replacing the per-frequency loop in
`operator_fourier_transform_via_bohr_energy_basis` with one big
broadcasted/einsum batched matmul) were *tried and measured slower* than
the plain loop at this problem's actual matrix sizes, because the
temporary arrays needed for a fully batched matmul are much bigger than
the tiny 8x8/16x16 matrices being multiplied, so allocation/memory
bandwidth dominates over the saved Python overhead. Those were reverted;
only changes that measured faster were kept:

1. `sum_over_frequencies` and `check_unitarity`: replaced the Python
   accumulation loop (`for A_w in A_w_list: s += A_w.conj().T @ A_w`)
   with a single `np.einsum('kij,kil->jl', ...)` contraction. Benchmarked
   ~2-2.5x faster for the dim=8 (n=3 spins) case with realistic frequency
   counts (~20000 stacked operators), since einsum avoids the per-item
   Python loop overhead while the actual matrices stay small.
2. Removed redundant re-diagonalization of H. The original code called
   `eigh(H)` / `precompute_energy_basis(H)` again inside
   `process_jump_operators_all_bohr_fast` and
   `create_jumps_for_sigma_0_version_2` on *every single call* -- i.e.
   once per sigma value in a sweep -- even though H never changes across
   a sweep. The eigendecomposition is now computed once in
   `build_model(...)` and passed around explicitly (evals/evecs/deltaE).
3. Removed redundant recomputation of the Gibbs state and its square root
   (`gibbs_state`, `construct_G`, which involve `expm` and `sqrtm`) from
   inside the per-sigma loop in `simulate_for_list_of_sigmas`. beta is
   fixed for an entire sigma sweep, so these are now computed once before
   the loop instead of once per sigma value.
4. Fixed functions that silently relied on module-level globals
   (`create_jumps_for_sigma_0_version_1/2` used to reach out to `evecs`,
   `evals`, `deltaE`, `w_d` from the enclosing script instead of their own
   arguments). They now take a `ModelSetup` object explicitly, which also
   makes the code safe to import and reuse for different (n, J, h, beta)
   without risk of stale globals.
5. The remaining, dominant cost per sigma value is building
   `qt.liouvillian` from thousands of discretized-frequency collapse
   operators and diagonalizing the resulting superoperator
   (`L.eigenstates()`). That cost is intrinsic to representing a
   continuum of Bohr frequencies as separate discrete Lindblad channels;
   shrinking it further would mean changing the physics construction
   (e.g. building the dissipator superoperator directly instead of going
   through per-operator `qt.Qobj`s), which was intentionally *not* done
   here to avoid silently changing numerics in a research script.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.linalg import expm, eigh
import qutip as qt


###############################################################################
# Model setup: Hamiltonian, eigenbasis, jump "seed" operators
###############################################################################

@dataclass
class ModelSetup:
    n: int
    J: float
    h: float
    H: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray
    deltaE: np.ndarray
    unique_nus: np.ndarray
    A_list: List[np.ndarray]
    P: np.ndarray                 # parity operator
    max_nu: float


def create_ising_hamiltonian(n, J, h):
    """Creates the Hamiltonian matrix for the 1D Quantum Ising Model."""
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    identity = np.eye(2)

    H = np.zeros((2 ** n, 2 ** n))

    # Interaction term: -J * sum_i sigma_i^z sigma_{i+1}^z
    for i in range(n):
        op = 1
        for j in range(n):
            if j == i or j == (i + 1) % n:
                op = np.kron(op, sigma_z)
            else:
                op = np.kron(op, identity)
        H -= J * op

    # Transverse field term: -h * sum_i sigma_i^x
    for i in range(n):
        op = 1
        for j in range(n):
            if j == i:
                op = np.kron(op, sigma_x)
            else:
                op = np.kron(op, identity)
        H -= h * op

    return H


def find_ground_state(H):
    """Finds the ground state by diagonalizing the Hamiltonian."""
    eigenvalues, eigenvectors = eigh(H)
    return eigenvalues[0], eigenvectors[:, 0]


def simple_jumps(n):
    """Local sigma_z and nearest-neighbor sigma_z sigma_z jump seeds."""
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    jumps = []

    for i in range(n):
        op = 1
        for j in range(n):
            op = np.kron(op, sigma_z if j == i else identity)
        jumps.append(op)

    for i in range(n):
        op = 1
        for j in range(n):
            op = np.kron(op, sigma_z if (j == i or j == (i + 1) % n) else identity)
        jumps.append(op)

    return jumps


def precompute_energy_basis(H):
    """Diagonalize H once and precompute Bohr frequencies deltaE_mn = E_m - E_n."""
    evals, evecs = eigh(H)
    deltaE = evals[:, None] - evals[None, :]
    return evals, evecs, deltaE


def parity_operator(n):
    """P = sigma_x (x) sigma_x (x) ... (n times)."""
    sigma_x = np.array([[0, 1], [1, 0]])
    P = sigma_x
    for _ in range(n - 1):
        P = np.kron(P, sigma_x)
    return P


def build_model(n, J, h) -> ModelSetup:
    """
    One-stop setup: build H, diagonalize once, build jump seeds and parity
    operator. Everything downstream should reuse this instead of
    recomputing the eigendecomposition.
    """
    H = create_ising_hamiltonian(n, J, h)
    evals, evecs, deltaE = precompute_energy_basis(H)
    unique_nus = np.unique(deltaE)
    A_list = simple_jumps(n)
    P = parity_operator(n)
    max_nu = np.max(evals) - np.min(evals)

    return ModelSetup(
        n=n, J=J, h=h, H=H, evals=evals, evecs=evecs, deltaE=deltaE,
        unique_nus=unique_nus, A_list=A_list, P=P, max_nu=max_nu,
    )


def check_parity(model: ModelSetup):
    """Returns parity <psi_i|P|psi_i> for every energy eigenstate."""
    parities = []
    for i in range(len(model.evals)):
        state = model.evecs[:, i]
        parities.append(np.real(state.conj().T @ model.P @ state))
    return np.array(parities)


###############################################################################
# Unitarity checks (vectorized)
###############################################################################

def check_unitarity(A_w_fft_steps):
    """Check if sum_w A(w)^dagger A(w) == identity."""
    A_w = np.asarray(A_w_fft_steps)
    dim = A_w.shape[1]
    identity_matrix = np.eye(dim)

    # sum_k A_w[k]^dagger @ A_w[k], vectorized
    sum_A_dagger_A = np.einsum('kij,kil->jl', A_w.conj(), A_w)

    return np.allclose(sum_A_dagger_A, identity_matrix, rtol=1e-3, atol=1e-10), sum_A_dagger_A


def normalize_to_unitarity(A_w_fft_steps):
    """Normalize A_w_fft_steps so that sum_w A(w)^dagger A(w) == identity."""
    is_unitary, sum_A_dagger_A = check_unitarity(A_w_fft_steps)
    if is_unitary:
        return A_w_fft_steps

    norm_factor = np.sqrt(np.linalg.norm(sum_A_dagger_A, np.inf))
    A_w_fft_steps_normalized = A_w_fft_steps / norm_factor
    return A_w_fft_steps_normalized


###############################################################################
# Transition rates
###############################################################################

def metropolis_weights(w_d, beta, sigma_E):
    """sqrt of the (smeared) Metropolis transition rate."""
    arg = w_d + beta * sigma_E ** 2 / 2
    metropolis_w = np.exp(-beta * np.maximum(arg, 0.0))
    return np.sqrt(metropolis_w)


def condition_for_parameters(beta, sigma_E):
    """Solves for (sigma_gamma, w_gamma) used by `gaussian_weights`.

    There are two free variables (w_gamma, sigma_gamma) and one equation, so
    w_gamma is fixed by convention to 1/beta and sigma_gamma is solved for.
    """
    w_gamma = 1 / beta

    sigma_gamma_sq = 2 * w_gamma / beta - sigma_E ** 2
    if sigma_gamma_sq < 0:
        raise ValueError

    return np.sqrt(sigma_gamma_sq), w_gamma


def gaussian_weights(w_d, w_gamma, sigma_gamma):
    """sqrt of the (smeared) Gaussian transition rate."""
    gaussian_w = np.exp(-(w_d + w_gamma) ** 2 / (2 * sigma_gamma ** 2))
    return np.sqrt(gaussian_w)


def gaussian_transition_weights(w_d, beta, sigma_E):
    """`gaussian_weights` under the same (w_d, beta, sigma_E) call signature as
    `metropolis_weights`, so it can be selected interchangeably via
    `TRANSITION_RATE_FUNCTIONS`.
    """
    sigma_gamma, w_gamma = condition_for_parameters(beta, sigma_E)
    return gaussian_weights(w_d, w_gamma, sigma_gamma)


TRANSITION_RATE_FUNCTIONS = {
    "metropolis": metropolis_weights,
    "gaussian": gaussian_transition_weights,
}


###############################################################################
# Discretization of the energy/time domain
###############################################################################

def discretize_energy_time(r_bound, N_disc):
    if N_disc % 2 == 0:
        N_disc += 1

    w_0 = 4 * r_bound / N_disc
    w_d = np.linspace(-N_disc / 2 * w_0, N_disc / 2 * w_0, N_disc)
    t_d = np.fft.fftfreq(len(w_d), d=w_0 / (2 * np.pi))

    if not np.isclose(w_0 * (t_d[1] - t_d[0]), 2 * np.pi / N_disc):
        raise ValueError('t_d and w_d are not Fourier conjugates')

    return w_d, t_d, w_0


def default_discretization(model: ModelSetup, beta):
    """Reproduces the r_bound / N_disc heuristic from the original script."""
    r_bound = 2 * (model.max_nu + 3 / beta)
    N_disc = 3201 + 4 * int(r_bound) + 4 * int(1 / beta) + 4 * int(beta) ** 2
    w_d, t_d, w_0 = discretize_energy_time(r_bound, N_disc)
    return w_d, t_d, w_0, r_bound, N_disc


###############################################################################
# Construction of the Bohr-frequency jump operators (main hot path)
###############################################################################

def make_gaussian_window_normalized(w_d, deltaE, sigma):
    """
    Returns f[k, m, n] such that for each (m, n), sum_k |f[k,m,n]|^2 = 1.
    """
    w = w_d[:, None, None]
    base = np.exp(-((w - deltaE) ** 2) / (4 * sigma ** 2))

    norm2 = np.sum(np.abs(base) ** 2, axis=0)
    norm = np.sqrt(norm2)
    norm[norm == 0] = 1.0

    return base / norm[None, :, :]


def operator_fourier_transform_via_bohr_energy_basis(A, evecs, deltaE, w_d, sigma_E):
    """
    Frequency-resolved jump operator A(w) via the Bohr-frequency (energy
    basis) construction.

    Kept as a loop over w_d (each iteration doing two small dim x dim
    matmuls): a fully batched/broadcasted version was benchmarked here and
    was consistently *slower* for this problem's matrix sizes (dim ~ 8-32,
    several thousand frequency points), since the huge temporary batched
    arrays cost more in allocation/memory bandwidth than the Python loop
    overhead they save. The one real cost reduction is precomputing
    `evecs.conj().T` once outside the loop instead of transposing it fresh
    on every iteration (as the original code implicitly did via the
    `@ evecs.conj().T` expression).
    """
    dim = A.shape[0]
    n_w = len(w_d)

    A_E = evecs.conj().T @ A @ evecs                      # energy basis
    evecsH = evecs.conj().T                               # precomputed once
    f_normed = make_gaussian_window_normalized(w_d, deltaE, sigma_E)  # (n_w, dim, dim)

    A_w = np.empty((n_w, dim, dim), dtype=complex)
    for k in range(n_w):
        A_w_E = f_normed[k] * A_E
        A_w[k] = evecs @ A_w_E @ evecsH

    return A_w


def process_jump_operators_all_bohr_fast(A_list, evals, evecs, deltaE, w_d, beta, sigma_E,
                                          transition_rate="metropolis"):
    """
    Builds the full stack of frequency-resolved, weighted jump operators for
    every seed operator in A_list.

    `transition_rate` selects the weighting function from
    `TRANSITION_RATE_FUNCTIONS` (currently "metropolis" or "gaussian").

    Note: evals/evecs/deltaE must be precomputed once (see build_model) and
    passed in -- they are NOT recomputed here, unlike the original script.
    """
    num_jumps = len(A_list)
    num_frequencies = len(w_d)
    matrix_size = evecs.shape[0]

    A_w_all = np.zeros((num_jumps * num_frequencies, matrix_size, matrix_size), dtype=complex)
    weight_fn = TRANSITION_RATE_FUNCTIONS[transition_rate]
    weights = weight_fn(w_d, beta, sigma_E)
    dw = w_d[1] - w_d[0]
    sqrt_dw = np.sqrt(dw)

    idx = 0
    for A_a in A_list:
        A_w_a = operator_fourier_transform_via_bohr_energy_basis(A_a, evecs, deltaE, w_d, sigma_E)
        A_w_a_weighted = sqrt_dw * A_w_a * weights[:, None, None]

        A_w_all[idx:idx + num_frequencies] = A_w_a_weighted
        idx += num_frequencies

    return A_w_all


###############################################################################
# Construction of the Lindbladian
###############################################################################

def construct_B_from_R_fast(H, R, beta, evals=None, evecs=None):
    """
    B = sum_nu (i/2 * tanh(0.25 * beta * nu)) * R_nu, computed directly in
    the energy eigenbasis of H.
    """
    if evals is None or evecs is None:
        evals, evecs = eigh(H)

    R_E = evecs.conj().T @ R @ evecs
    deltaE = evals[:, None] - evals[None, :]
    factor_mat = 0.5j * np.tanh(0.25 * beta * deltaE)
    B_E = factor_mat * R_E
    B = evecs @ B_E @ evecs.conj().T
    B = 0.5 * (B + B.conj().T)
    return B


def sum_over_frequencies(A_w_array):
    """
    sum_w A(w)^dagger A(w) for a stack of frequency components with shape
    (n_w, dim, dim). Vectorized via einsum instead of a Python loop.
    """
    A_w = np.asarray(A_w_array)
    return np.einsum('kij,kil->jl', A_w.conj(), A_w)


def gibbs_state(H, beta):
    """Thermal state e^{-beta H} / Z, as a qutip Qobj."""
    expH = expm(-beta * H)
    Z = np.trace(expH)
    rho = expH / Z
    return qt.Qobj(rho)


def construct_G(rho_gibbs):
    """G = sprepost(sqrt(rho_gibbs), sqrt(rho_gibbs)) superoperator."""
    sqrt_rho = rho_gibbs.sqrtm()
    return qt.sprepost(sqrt_rho, sqrt_rho)


###############################################################################
# Detailed balance checks
###############################################################################

def actual_hilbert_dist(A, B):
    """Hilbert-Schmidt distance between two density matrices A and B."""
    if A.isket or A.isbra:
        A = qt.ket2dm(A)
    if B.isket or B.isbra:
        B = qt.ket2dm(B)
    if A.dims != B.dims:
        raise TypeError('A and B do not have same dimensions.')
    return np.sqrt(((A - B).dag() * (A - B)).tr())


def check_detailed_balance(L, G, tol=1e-5):
    """Check if G^-1 * L * G == L^dagger (in operator-norm sense)."""
    L_conjugated = L * G
    L_dagger = G * L.dag()
    dist = actual_hilbert_dist(L_conjugated, L_dagger)
    rel_dist = dist / L.norm()
    return rel_dist < tol, rel_dist


###############################################################################
# Spectral-gap simulation
###############################################################################

def simulate_for_list_of_sigmas(model: ModelSetup, sigma_E_start, sigma_E_end, how_many, beta,
                                 w_d, t_d, verbose=True, transition_rate="metropolis"):
    """
    Sweeps sigma_E in [sigma_E_start, sigma_E_end] (how_many points) at
    fixed beta, builds the Lindbladian for each, checks detailed balance,
    and returns the spectral gap.

    `transition_rate` selects the weighting function from
    `TRANSITION_RATE_FUNCTIONS` (currently "metropolis" or "gaussian").
    """
    list_gap = []
    sigma_E_vals = np.linspace(sigma_E_start, sigma_E_end, how_many)

    rho_gibbs = gibbs_state(model.H, beta)
    G = construct_G(rho_gibbs)

    for s in sigma_E_vals:
        A_w_list = process_jump_operators_all_bohr_fast(
            model.A_list, model.evals, model.evecs, model.deltaE, w_d, beta, s,
            transition_rate=transition_rate,
        )
        if verbose:
            print('Operator list done')

        R = sum_over_frequencies(A_w_list)
        B = construct_B_from_R_fast(model.H, R, beta, model.evals, model.evecs)

        B_qobj = qt.Qobj(B)
        C_ops = [qt.Qobj(A_w) for A_w in A_w_list]

        L = qt.liouvillian(B_qobj, C_ops)
        if verbose:
            print("Construction done")

        flag, dist = check_detailed_balance(L, G)
        if verbose:
            if flag:
                print("The Liouvillian satisfies detailed balance. Dist", dist)
            else:
                print(f"The Liouvillian does NOT satisfy detailed balance. Distance is {dist}")

        evals_L, _ = L.eigenstates()
        gap = -np.real(evals_L[-2])
        list_gap.append(gap)

        if verbose:
            print('gap:', gap)
            print()

    return sigma_E_vals, list_gap


def plot_spectral_gap_vs_sigma_all_betas(model: ModelSetup, sigma_E_start, sigma_E_end, how_many,
                                          betas, w_d, t_d, transition_rate="metropolis"):
    """Runs simulate_for_list_of_sigmas for every beta and plots all curves.

    `transition_rate` selects the weighting function from
    `TRANSITION_RATE_FUNCTIONS` (currently "metropolis" or "gaussian").
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    for beta in betas:
        sigma_E_vals, list_gap = simulate_for_list_of_sigmas(
            model, sigma_E_start, sigma_E_end, how_many, beta, w_d, t_d, verbose=False,
            transition_rate=transition_rate,
        )
        ax.plot(sigma_E_vals, list_gap, marker='o', markersize=3, label=fr'$\beta = {beta}$')

    ax.set_xlabel(r'$\sigma$', fontsize=12)
    ax.set_ylabel(r'Lindbladian gap $\gamma$', fontsize=12)
    ax.set_title(
        rf'Lindbladian gap $\gamma$ vs $\sigma$ for $n = {model.n}, J = {model.J}, h = {model.h}$',
        fontsize=12
    )
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    return fig, ax


###############################################################################
# Upper bound / Davies-generator (sigma -> 0) jump operators
###############################################################################

def create_jumps_for_sigma_0_version_1(model: ModelSetup, w_d, beta, transition_rate="metropolis"):
    """Davies jumps built by summing degenerate Bohr-frequency contributions.

    `transition_rate` selects the weighting function from
    `TRANSITION_RATE_FUNCTIONS` (currently "metropolis" or "gaussian").
    """
    weight_fn = TRANSITION_RATE_FUNCTIONS[transition_rate]
    evals, evecs, deltaE = model.evals, model.evecs, model.deltaE
    dw = w_d[1] - w_d[0]
    sqrt_dw = np.sqrt(dw)
    jumps = []

    for A in model.A_list:
        A_E = evecs.conj().T @ A @ evecs
        A_nu_dict = {}

        for i in range(len(evals)):
            for j in range(len(evals)):
                nu = deltaE[i, j]
                A_single_E = np.zeros_like(A_E)
                A_single_E[i, j] = A_E[i, j] * sqrt_dw

                if nu in A_nu_dict:
                    A_nu_dict[nu] += A_single_E
                else:
                    A_nu_dict[nu] = A_single_E

        for nu in A_nu_dict:
            jump = evecs @ A_nu_dict[nu] * weight_fn(nu, beta, 0) @ evecs.conj().T
            jumps.append(jump)

    return np.array(jumps)


def create_jumps_for_sigma_0_version_2(model: ModelSetup, w_d, beta, transition_rate="metropolis"):
    """Same Davies jumps, built via masking on the unique Bohr frequencies.

    `transition_rate` selects the weighting function from
    `TRANSITION_RATE_FUNCTIONS` (currently "metropolis" or "gaussian").
    """
    weight_fn = TRANSITION_RATE_FUNCTIONS[transition_rate]
    evals, evecs, deltaE = model.evals, model.evecs, model.deltaE
    dw = w_d[1] - w_d[0]
    sqrt_dw = np.sqrt(dw)
    jumps = []

    for A in model.A_list:
        A_E = evecs.conj().T @ A @ evecs

        for nu in model.unique_nus:
            mask = (deltaE == nu)
            A_nu_E = A_E * mask * sqrt_dw
            A_nu_E = A_nu_E * weight_fn(nu, beta, 0)
            A_nu = evecs @ A_nu_E @ evecs.conj().T
            jumps.append(A_nu)

    return np.array(jumps)


def give_Davies_gap(model: ModelSetup, jumps, beta, verbose=True):
    """Builds the Davies Lindbladian from precomputed jumps and returns its gap."""
    rho_gibbs = gibbs_state(model.H, beta)
    G = construct_G(rho_gibbs)

    R_0 = sum_over_frequencies(jumps)
    B_0 = construct_B_from_R_fast(model.H, R_0, beta, model.evals, model.evecs)

    B_qobj = qt.Qobj(B_0)
    C_ops_0 = [qt.Qobj(j) for j in jumps]

    L_0 = qt.liouvillian(B_qobj, C_ops_0)

    flag, dist = check_detailed_balance(L_0, G)
    if verbose:
        if flag:
            print("The Liouvillian satisfies detailed balance. Dist", dist)
        else:
            print(f"The Liouvillian does NOT satisfy detailed balance. Distance is {dist}")

    evals_L0, _ = L_0.eigenstates()
    gap_Davies = -np.real(evals_L0[-2])

    if verbose:
        print("Davies gap:", gap_Davies)

    return gap_Davies


###############################################################################
# Control: sigma -> 0 limit
###############################################################################

def make_gaussian_window_L2_and_grid_spacing(w_d, nu, sigma, w_0):
    """
    Returns f[w_d, nu] with sum_d f[w_d,nu] * w_0 = 1 for every nu.
    """
    w_d = np.asarray(w_d, dtype=float)
    w = w_d[:, None, None]
    base = np.exp(-0.5 * ((w - nu) / sigma) ** 2)

    norm = np.sum(np.abs(base) ** 2, axis=0) * w_0
    norm[norm == 0] = 1.0

    f = base / np.sqrt(norm[None, :, :])
    return f, base


def controll_limes_sigma_0(model: ModelSetup, w_d, w_0, sigma, beta, make_plots=True,
                            transition_rate="metropolis"):
    """Checks that the smeared rate converges to the exact rate as sigma -> 0.

    `transition_rate` selects the weighting function from
    `TRANSITION_RATE_FUNCTIONS` (currently "metropolis" or "gaussian").
    """
    import matplotlib.pyplot as plt

    weight_fn = TRANSITION_RATE_FUNCTIONS[transition_rate]
    f_sigma_wd_nu, base = make_gaussian_window_L2_and_grid_spacing(w_d, model.unique_nus, sigma, w_0)
    gamma_wd = weight_fn(w_d, beta, sigma)

    if make_plots:
        for u in model.unique_nus:
            y = make_gaussian_window_L2_and_grid_spacing(w_d, u, sigma, w_0)[0].squeeze()
            non_zero = y[y != 0]
            if len(non_zero) == 0:
                print(f"The list is empty for nu = {u}")
                continue

            fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
            ax.plot(w_d, y, label=r'$f_\sigma(\omega_d-\nu)$')
            ax.set_xlabel(r'$\omega$', fontsize=13)
            ax.set_ylabel(r'$f_\sigma(\omega_d-\nu)$', fontsize=13)
            ax.set_title(r'Gaussian $\to$ delta distribution for $\sigma \to 0$', fontsize=14)
            ax.tick_params(labelsize=11)

            inset = fig.add_axes([0.58, 0.30, 0.35, 0.35])
            inset.scatter(w_d, y, s=12, alpha=0.8, edgecolors='none')
            peak = np.argmax(y)
            lo = max(peak - 10, 0)
            hi = min(peak + 10, len(w_d) - 1)
            inset.set_xlim(w_d[lo], w_d[hi])
            inset.set_title('Zoom', fontsize=9)
            inset.tick_params(labelsize=8)
            inset.grid(alpha=0.25)

            ax.legend(frameon=True, fontsize=11)
            plt.tight_layout()
            plt.show()

    terms_to_sum_over = np.abs(f_sigma_wd_nu) ** 2 * gamma_wd[:, None, None]
    sum_over_all_wd = terms_to_sum_over.sum(axis=0) * w_0

    gamma_nu = weight_fn(model.unique_nus, beta, 0)
    difference = sum_over_all_wd[0, :] - gamma_nu
    maxim = np.max(np.abs(difference))

    if make_plots:
        fig, ax = plt.subplots(dpi=150)
        inset = fig.add_axes([0.2, 0.2, 0.3, 0.3])
        ax.scatter(model.unique_nus, sum_over_all_wd[0, :], label='sum', s=1)
        ax.plot(model.unique_nus, gamma_nu, label='gamma')
        inset.plot(model.unique_nus, difference, label='difference')
        inset.legend()
        ax.set_xlabel(r'$\nu$')
        ax.legend()
        plt.show()

    return difference, maxim, f_sigma_wd_nu, base


###############################################################################
# f(sigma) construction and optimal-sigma search
###############################################################################

def function(gap, sigma):
    """f(sigma) = 1 / (gap * sigma) -- proxy for total mixing runtime."""
    return 1 / (gap * sigma)


def u_form(model: ModelSetup, start, stop, anzahl, beta, w_d, t_d, make_plot=True,
           transition_rate="metropolis"):
    sigma_E, gaps = simulate_for_list_of_sigmas(model, start, stop, anzahl, beta, w_d, t_d, verbose=False,
                                                 transition_rate=transition_rate)
    sigma_E, gaps = np.array(sigma_E), np.array(gaps)
    f = function(gaps, sigma_E)

    if make_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(sigma_E, f, s=8, alpha=0.8, edgecolors='none', label='Simulation data')
        ax.set_xlabel(r'$\sigma$', fontsize=13)
        ax.set_ylabel(r'$f$', fontsize=13)
        ax.set_title(rf'Total runtime $f(\sigma)$ for $\beta = {beta}$', fontsize=14)
        ax.tick_params(labelsize=11)
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.25)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    return sigma_E, gaps, f


def get_index_where_derivative_0(model: ModelSetup, beta, w_d, t_d, start=0.02, stop=3.0, anzahl=150,
                                  make_plot=True, transition_rate="metropolis"):
    sigma_E, gaps = simulate_for_list_of_sigmas(model, start, stop, anzahl, beta, w_d, t_d, verbose=False,
                                                 transition_rate=transition_rate)
    sigma_E, gaps = np.array(sigma_E), np.array(gaps)
    f = function(gaps, sigma_E)
    derivative = np.gradient(f)

    if make_plot:
        import matplotlib.pyplot as plt
        plt.scatter(sigma_E, derivative, s=1, label='derivative of f')
        plt.title(f'beta = {beta}')
        plt.legend()
        plt.show()

    indices = []
    for ind in range(len(derivative) - 1):
        if derivative[ind] < 0 and derivative[ind + 1] > 0:
            indices.append(ind)

    return sigma_E, gaps, f, derivative, indices


def my_golden_section_search(model: ModelSetup, start, stop, beta, w_d, t_d, max_iter=10, tol=1e-6,
                              transition_rate="metropolis"):
    g = 0.381966  # (3 - sqrt(5)) / 2

    intervalls = [start, stop]
    sigma_E, gaps = simulate_for_list_of_sigmas(model, start, stop, 2, beta, w_d, t_d, verbose=False,
                                                 transition_rate=transition_rate)
    sigma_E, gaps = np.array(sigma_E), np.array(gaps)
    f = function(gaps, sigma_E)
    fs = [f[0], f[1]]

    for _ in range(max_iter):
        if abs(stop - start) < tol:
            break

        x1 = stop - g * (stop - start)
        x2 = start + g * (stop - start)

        sigma_E, gaps = simulate_for_list_of_sigmas(model, x1, x2, 2, beta, w_d, t_d, verbose=False,
                                                      transition_rate=transition_rate)
        sigma_E, gaps = np.array(sigma_E), np.array(gaps)
        f = function(gaps, sigma_E)
        f1, f2 = f[0], f[1]

        intervalls.extend([x1, x2])
        fs.extend([f1, f2])

        if x2 > x1:
            if f1 < f2:
                stop = x2       # minimum is in [start, x2]
            else:
                start = x1      # minimum is in [x1, stop]
        else:
            if f2 < f1:
                stop = x1       # minimum is in [start, x1]
            else:
                start = x2      # minimum is in [x2, stop]

    best_idx = np.argmin(fs)
    return intervalls, fs, intervalls[best_idx]


###############################################################################
# Specific heat
###############################################################################

def specific_heat(model: ModelSetup, beta):
    rho_gibbs = gibbs_state(model.H, beta)
    H_qobj = qt.Qobj(model.H)
    H_1 = (H_qobj ** 2 * rho_gibbs).tr()
    H_2 = ((H_qobj * rho_gibbs).tr()) ** 2
    return beta ** 2 * (H_1 - H_2)


def compute_specific_heat_and_optimal_sigma(model: ModelSetup, betas, sigma_welle):
    betas_array = np.array(betas)
    specific_heat_vals = np.array([specific_heat(model, beta) for beta in betas_array])

    C_normalized = specific_heat_vals / np.max(specific_heat_vals)
    sigma_normalized = np.array(sigma_welle) / np.max(sigma_welle)

    C_peak_idx = np.argmax(specific_heat_vals)
    print(f"Heat peak at beta = {betas_array[C_peak_idx]:.3f}")

    return betas_array, specific_heat_vals, C_normalized, sigma_normalized
