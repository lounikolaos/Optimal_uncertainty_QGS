"""
definitions.py

All function/class definitions for the quantum Gibbs sampler:
jump-operator construction, the three Lindbladian constructions
(time domain, frequency domain, Anthony's construction), convergence-time
evolution, the convergence sweep runner, and plotting helpers.

"""

from __future__ import annotations
import sys
import time
import itertools
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import random
import pandas as pd
import scipy.linalg
from scipy.linalg import eigh, expm
import matplotlib.pyplot as plt

sys.path.append('..')
from gibbs_state_prep.helpers import ising_1d, heisenberg_1d, compute_trace_distance, I, X, Y, Z, kron, kron_n
from gibbs_state_prep.algorithm.preparation import IntegrationGrid
from gibbs_state_prep.exact.gibbs_state import GibbsState


def simple_jumps(n):
    """Local Z_i jump operators, shape (n, 2**n, 2**n), normalized so that
    ||sum_i A_i^dagger A_i|| = 1.
    """
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    jumps = []
    for i in range(n):
        op = 1
        for j in range(n):
            op = np.kron(op, sigma_z if j == i else identity)
        jumps.append(op)

    return np.array(jumps)


def normalized_simple_jumps(n):
    """Local Z_i jump operators, shape (n, 2**n, 2**n), normalized so that
    ||sum_i A_i^dagger A_i|| = 1.
    """
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    jumps = []
    for i in range(n):
        op = 1
        for j in range(n):
            op = np.kron(op, sigma_z if j == i else identity)
        jumps.append(op)
    jumps = np.array(jumps) / np.sqrt(n)

    R = np.einsum('mij,mik->jk', jumps.conj(), jumps)
    normalization = np.linalg.norm(R, ord=2)
    if not np.isclose(normalization, 1.0):
        raise ValueError(f"||sum_i A_i^dagger A_i|| = {normalization}, expected 1")

    return jumps


def _fft_frequency_to_time(nu_grid: np.ndarray, dnu: float, g_of_nu: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """f(t) = (1/2pi) * integral g(nu) e^{-i t nu} dnu, computed via FFT."""
    g_nu = np.asarray(g_of_nu(nu_grid), dtype=complex)
    g_unshift = np.fft.ifftshift(g_nu)
    f_unshift = np.fft.fft(g_unshift)
    f_t = np.fft.fftshift(f_unshift)
    f_t *= (dnu / (2.0 * np.pi))
    return f_t


def check_array_closeness(a, b, atol=1e-4, rtol=1e-5):
    if a.shape != b.shape:
        raise ValueError("Arrays must have the same shape.")
    close_mask = np.isclose(a, b, atol=atol, rtol=rtol)
    diff = a - b
    return {
        "close_all": bool(close_mask.all()),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "close_mask": close_mask,
        "mismatch_indices": np.argwhere(~close_mask),
    }


def kraus_ops_to_superoperator(kraus_ops: np.ndarray) -> np.ndarray:
    """sum_m kron(conj(A_m), A_m) -- equivalent to qutip.kraus_to_super under column-stacking
    vectorization (vec(rho)[i + j*d] = rho[i, j]), computed for all Kraus operators in one shot.
    """
    A = np.asarray(kraus_ops)
    d = A.shape[-1]
    S4 = np.einsum('mij,mkl->ikjl', A.conj(), A)
    return S4.reshape(d * d, d * d)


def superop_spre_plus_spostdag(G: np.ndarray) -> np.ndarray:
    """Superoperator for rho -> G@rho + rho@G^dagger, i.e. qutip.spre(G) + qutip.spost(G.dag())."""
    d = G.shape[-1]
    Id = np.eye(d, dtype=complex)
    return np.kron(Id, G) + np.kron(G.conj(), Id)


def get_eig(hamiltonian: np.ndarray, eig_cache: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Shared Hamiltonian-eigendecomposition cache used by all three constructions."""
    if eig_cache is not None and "evals" in eig_cache:
        return eig_cache["evals"], eig_cache["evecs"]
    evals, evecs = eigh(hamiltonian)
    if eig_cache is not None:
        eig_cache["evals"], eig_cache["evecs"] = evals, evecs
    return evals, evecs


def _bump_window(S: float) -> Callable[[np.ndarray], np.ndarray]:
    """Smooth compactly-supported window: 1 for |nu|<=S/2, 0 for |nu|>=S, smooth in between."""
    if S <= 0:
        raise ValueError("S must be positive for smooth jump window.")
    inner, outer = 0.5 * S, S
    width = outer - inner

    def _flat_step(x):
        out = np.zeros_like(x, dtype=float)
        mask = x > 0.0
        out[mask] = np.exp(-1.0 / x[mask])
        return out

    def w(nu):
        abs_nu = np.abs(np.asarray(nu, dtype=float))
        out = np.zeros_like(abs_nu, dtype=float)
        inside, outside = abs_nu <= inner, abs_nu >= outer
        transition = (~inside) & (~outside)
        out[inside] = 1.0
        if np.any(transition):
            z = (abs_nu[transition] - inner) / width
            hz, h1mz = _flat_step(z), _flat_step(1.0 - z)
            denom = hz + h1mz
            vals = np.zeros_like(z)
            safe = denom > 0.0
            vals[safe] = h1mz[safe] / denom[safe]
            out[transition] = vals
        return out

    return w


def q_metropolis_bumped(nu: np.ndarray, beta: float, S: float, sigma: float = 2.0) -> np.ndarray:
    """Paper Eq. (3.20) Metropolis-type filter: exp(-sqrt(1+(beta*nu)^2)/4) * bump(nu/S)."""
    nu = np.asarray(nu, dtype=float)
    if beta < 0:
        raise ValueError("beta should be nonnegative.")
    envelope = np.exp(-0.25 * np.sqrt(1.0 + (beta * nu) ** 2))

    return envelope * _bump_window(S)(nu)



def q_gaussian_bumped(nu, beta, S=10.0, sigma=2.0):
    gaussian = np.exp(-(beta * nu)**2 / 8)
    return gaussian * _bump_window(S)(nu)


def q_gaussian(nu, beta, S=10.0, sigma=2.0):
    return np.exp(-(beta * nu)**2 / 8)


def q_gaussian_bumped_sigma(nu, beta, S=10.0, sigma=2.0):
    gaussian = np.exp(-(nu)**2 / (8*sigma**2))
    return gaussian * _bump_window(S)(nu)


def q_gaussian_sigma(nu, beta, S=10.0, sigma=2.0):
    return np.exp(-(nu)**2 / (8 *sigma**2))



Q_FILTER_FUNCTIONS = {
    "metropolis_bumped": q_metropolis_bumped,
    "gaussian_bumped": q_gaussian_bumped,
    "gaussian": q_gaussian,
    "gaussian_bumped_sigma": q_gaussian_bumped_sigma,
    "gaussian_sigma": q_gaussian_sigma,
}


def compute_filter_time_domain(beta: float, S: float, sigma: float, grid: IntegrationGrid,
                                q_filter_func: Callable) -> np.ndarray:
    """f_a(t) = (1/2pi) * integral q(nu) * exp(-beta*nu/4) * exp(-i t nu) dnu, via FFT."""
    def g_kms(nu):
        q_nu = q_filter_func(nu, beta, S, sigma).astype(complex)
        #exponent = -(beta / 4.0) * nu
        exponent = -(beta / 4.0) * nu
        w_nu = np.exp(np.clip(exponent, -700.0, 700.0))
        out = q_nu * w_nu
        out[~np.isfinite(out)] = 0.0
        return out
    return _fft_frequency_to_time(grid.nu_grid, grid.dnu, g_kms)


def batch_time_evolve(ops_eig: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                       t_grid: np.ndarray) -> np.ndarray:
    """Time-evolve a batch of operators (already expressed in the eigenbasis of H) at every
    time in t_grid simultaneously.

    ops_eig: (n_ops, d, d) = V^dagger @ A @ V for each operator A
    returns: (n_t, n_ops, d, d) array of A(t) = e^{iHt} A e^{-iHt} in the ORIGINAL basis
    """
    phases_all = np.exp(1j * np.outer(t_grid, eigenvalues))            # (n_t, d)
    evolved_eig = (phases_all[:, None, :, None]
                   * ops_eig[None, :, :, :]
                   * phases_all.conj()[:, None, None, :])              # (n_t, n_ops, d, d)
    evolved = np.einsum('ab,tobc,dc->toad', eigenvectors, evolved_eig, eigenvectors.conj())
    return evolved


def build_L_alpha(coupling_ops: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                   grid: IntegrationGrid, beta: float, S: float, sigma: float,
                   q_filter_func: Callable) -> np.ndarray:
    """L_alpha = integral f(s) e^{iHs} A_alpha e^{-iHs} ds, for every coupling operator at once."""
    ops_eig = eigenvectors.conj().T @ coupling_ops @ eigenvectors        # (n_ops, d, d)
    A_t = batch_time_evolve(ops_eig, eigenvalues, eigenvectors, grid.t_grid)  # (n_t, n_ops, d, d)
    f_t = compute_filter_time_domain(beta, S, sigma, grid, q_filter_func)     # (n_t,)
    weighted = A_t * f_t[:, None, None, None] * grid.dt
    return weighted.sum(axis=0) 


def plot_comp_of_L_alpha(coupling_ops: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                          grid: IntegrationGrid, beta_list: list[float], S_list: list[float],
                          sigma_list: list[float], q_filter_func: Callable,
                          labels: list[str]) -> None:
    """L_alpha = f(s) e^{iHs} A_alpha e^{-iHs} ds , for every coupling operator at once.
    Overlays multiple parameter sets (e.g. different betas) on the same subplots,
    one subplot per coupling operator.
    """
    t = grid.t_grid
    ops_eig = eigenvectors.conj().T @ coupling_ops @ eigenvectors        # (n_ops, d, d)
    A_t = batch_time_evolve(ops_eig, eigenvalues, eigenvectors, grid.t_grid)  # (n_t, n_ops, d, d)
    n_ops = coupling_ops.shape[0]

    fig, axes = plt.subplots(1, n_ops, figsize=(8 * n_ops, 6), sharey=True)

    for j, (beta, S, sigma, label) in enumerate(zip(beta_list, S_list, sigma_list, labels)):
        f_t = compute_filter_time_domain(beta, S, sigma, grid, q_filter_func)     # (n_t,)
        weighted = A_t * f_t[:, None, None, None] * grid.dt
        Ls = np.swapaxes(weighted, 0, 1)
        norms = np.linalg.norm(Ls, ord=np.inf, axis=(-2, -1))  # shape (n_ops, n_t)

        for i, ax in enumerate(axes):
            ax.scatter(t, norms[i], s=1, color=f'C{j}', label=label)

    for i, ax in enumerate(axes):
        ax.set_xlabel('time')
        ax.set_title(f'Supremum norm of $L_{{{i+1}}}(t)$ over time')
        ax.grid(True, alpha=0.3)
        ax.legend(markerscale=8)

    axes[0].set_ylabel(r'$\|L_\alpha(t)\|_\infty$')
    plt.tight_layout()
    plt.show()


def validate_fast_linear_algebra():
    rng = np.random.default_rng(0)

    # 1) Kraus map -> superoperator, vs. direct application of the Kraus map
    d, m = 5, 4
    A = rng.standard_normal((m, d, d)) + 1j * rng.standard_normal((m, d, d))
    rho = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho_prime_direct = sum(A[k] @ rho @ A[k].conj().T for k in range(m))
    S = kraus_ops_to_superoperator(A)
    rho_prime_from_S = (S @ rho.reshape(-1, 1, order='F')).reshape(d, d, order='F')
    assert np.allclose(rho_prime_direct, rho_prime_from_S), "Kraus->superoperator mismatch"

    # 2) spre(G) + spost(G^dagger), vs. G@rho + rho@G^dagger directly
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    direct = G @ rho + rho @ G.conj().T
    U = superop_spre_plus_spostdag(G)
    from_U = (U @ rho.reshape(-1, 1, order='F')).reshape(d, d, order='F')
    assert np.allclose(direct, from_U), "spre+spost formula mismatch"

    # 3) batched eigenbasis time evolution, vs. a plain per-time-step loop
    H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = H + H.conj().T
    evals, evecs = eigh(H)
    n_ops, n_t = 3, 12
    ops = rng.standard_normal((n_ops, d, d)) + 1j * rng.standard_normal((n_ops, d, d))
    t_grid = np.linspace(0, 4, n_t)
    ops_eig = evecs.conj().T @ ops @ evecs

    ref = np.zeros((n_t, n_ops, d, d), dtype=complex)
    for ti, t in enumerate(t_grid):
        phases = np.exp(1j * evals * t)
        for o in range(n_ops):
            evolved_eig = (phases[:, None] * ops_eig[o]) * phases.conj()[None, :]
            ref[ti, o] = evecs @ evolved_eig @ evecs.conj().T

    fast = batch_time_evolve(ops_eig, evals, evecs, t_grid)
    assert np.allclose(ref, fast), "batched time evolution mismatch"

    print("All fast linear-algebra identities match their reference implementations.")


def prefactor_to_build_X(sigma_E, s):
    """Gaussian envelope used to window the time-evolved jump operators into X_alpha(s)."""
    return np.sqrt(sigma_E * np.sqrt(2 / np.pi)) * np.exp(-sigma_E**2 * s**2)


def build_X_s(L_alpha_ops: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
              grid: IntegrationGrid, sigma: float) -> np.ndarray:
    """X'_alpha(s) = sqrt(sigma*sqrt(2/pi)) * exp(-sigma^2 s^2) * e^{iHs} L_alpha e^{-iHs}."""
    ops_eig = eigenvectors.conj().T @ L_alpha_ops @ eigenvectors
    A_t = batch_time_evolve(ops_eig, eigenvalues, eigenvectors, grid.t_grid)  # (n_t, n_ops, d, d)
    window = prefactor_to_build_X(sigma, grid.t_grid)                             # (n_t,)
    return A_t * window[:, None, None, None]                             # (n_t, n_ops, d, d)


def plot_builded_X_s(L_alpha_ops_list: list[np.ndarray], eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                      grid: IntegrationGrid, sigma_list: list[float], labels: list[str]) -> None:
    """X'_alpha(s) = sqrt(sigma*sqrt(2/pi)) * exp(-sigma^2 s^2) * e^{iHs} L_alpha e^{-iHs}.
    Overlays multiple parameter sets (e.g. different betas) on the same subplots,
    one subplot per coupling operator.
    """
    t = grid.t_grid
    n_ops = L_alpha_ops_list[0].shape[0]

    fig, axes = plt.subplots(1, n_ops, figsize=(8 * n_ops, 6), sharey=True)

    for j, (L_alpha_ops, sigma, label) in enumerate(zip(L_alpha_ops_list, sigma_list, labels)):
        ops_eig = eigenvectors.conj().T @ L_alpha_ops @ eigenvectors
        A_t = batch_time_evolve(ops_eig, eigenvalues, eigenvectors, grid.t_grid)  # (n_t, n_ops, d, d)
        window = prefactor_to_build_X(sigma, grid.t_grid)                        # (n_t,)
        X_s = A_t * window[:, None, None, None]                                  # (n_t, n_ops, d, d)
        Xs = np.swapaxes(X_s, 0, 1)
        norms = np.linalg.norm(Xs, ord=np.inf, axis=(-2, -1))  # shape (n_ops, n_t)

        for i, ax in enumerate(axes):
            ax.scatter(t, norms[i], s=1, color=f'C{j}', label=label)

    for i, ax in enumerate(axes):
        ax.set_xlabel('time')
        ax.set_title(f'Supremum norm of $X_{{{i+1}}}(t)$ over time')
        ax.grid(True, alpha=0.3)
        ax.legend(markerscale=8)

    axes[0].set_ylabel(r'$\|X_\alpha(t)\|_\infty$')
    plt.tight_layout()
    plt.show()


def build_G_term(L_ops: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                  grid: IntegrationGrid, beta: float, sigma: float) -> np.ndarray:
    """G = -sum_alpha integral g(t) e^{iHt} L_alpha^dagger L_alpha e^{-iHt} dt."""
    L_dag_L = np.matmul(L_ops.conj().transpose(0, 2, 1), L_ops)          # (n_ops, d, d)
    ops_eig = eigenvectors.conj().T @ L_dag_L @ eigenvectors
    A_t = batch_time_evolve(ops_eig, eigenvalues, eigenvectors, grid.t_grid)  # (n_t, n_ops, d, d)

    def g_nu(nu):
        return np.exp(-nu**2 / (8 * sigma**2)) / (1 + np.exp(beta * nu / 2))

    g_t = _fft_frequency_to_time(grid.nu_grid, grid.dnu, g_nu)           # (n_t,)
    weighted = A_t * g_t[:, None, None, None] * grid.dt
    G_alpha = weighted.sum(axis=0)                                       # (n_ops, d, d)
    return -G_alpha.sum(axis=0)                                          # (d, d)



def build_lindbladian_time_domain(hamiltonian: np.ndarray, beta: float, sigma: float, S: float,
                                   grid: IntegrationGrid, coupling_ops: np.ndarray,
                                   q_filter: str = "metropolis",
                                   eig_cache: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (Lindbladian, rho_gibbs) as dense NumPy arrays.

    Lindbladian acts on vec(rho) with column-stacking (Fortran-order) convention, matching
    evolve_until_converge._apply_lindbladian below.
    """
    eigenvalues, eigenvectors = get_eig(hamiltonian, eig_cache)
    q_filter_func = Q_FILTER_FUNCTIONS[q_filter]

    L_alpha = build_L_alpha(coupling_ops, eigenvalues, eigenvectors, grid, beta, S, sigma, q_filter_func)
    X_s = build_X_s(L_alpha, eigenvalues, eigenvectors, grid, sigma)     # (n_t, n_ops, d, d)

    d = hamiltonian.shape[0]
    kraus_ops = (X_s * np.sqrt(grid.dt)).reshape(-1, d, d)
    phi_super = kraus_ops_to_superoperator(kraus_ops)

    G_term = build_G_term(L_alpha, eigenvalues, eigenvectors, grid, beta, sigma)
    U_super = superop_spre_plus_spostdag(G_term)

    Lindbladian = phi_super + U_super
    rho_gibbs = GibbsState(hamiltonian, beta).prepare()
    return Lindbladian, rho_gibbs


def filter_function_nu(nu: np.ndarray, beta: float, S: float, sigma: float, q_filter_func: Callable) -> np.ndarray:
    """f(nu) = q(nu, beta, S, sigma) * exp(-beta*nu/4), evaluated directly (no FFT)."""
    q_nu = q_filter_func(nu, beta, S, sigma).astype(complex)
    exponent = -(beta / 4.0) * nu
    w_nu = np.exp(np.clip(exponent, -700.0, 700.0))
    out = q_nu * w_nu
    out[~np.isfinite(out)] = 0.0
    return out


def check_filter_kms_condition(nu: np.ndarray, beta: float, S: float, sigma: float,
                                q_filter_func: Callable, atol: float = 1e-4, rtol: float = 1e-5) -> dict:
    """Checks the detailed-balance/KMS condition conj(f(nu)) = f(-nu) * exp(-beta*nu/2), where
    f = filter_function_nu(nu, beta, S, sigma, q_filter_func).
    """
    nu = np.asarray(nu, dtype=float)
    lhs = filter_function_nu(nu, beta, S, sigma, q_filter_func).conj()
    rhs = filter_function_nu(-nu, beta, S, sigma, q_filter_func) * np.exp(-beta * nu / 2.0)
    return check_array_closeness(lhs, rhs, atol=atol, rtol=rtol)


def _prefactor_to_build_G(nu1, nu2, sigma, beta):
    p1 = np.exp(-(nu1 - nu2) ** 2 / (8 * sigma ** 2))
    p2 = 1.0 / (1.0 + np.exp(beta * (nu2 - nu1) / 2.0))
    return p1 * p2


def _prefactor_for_phi(nu1, nu2, sigma):
    return np.exp(-(nu1 - nu2) ** 2 / (8 * sigma ** 2))


def bohr_frequency_components(coupling_ops: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                               nu_round: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Decomposes each coupling operator into its Bohr-frequency components A^alpha_nu, i.e.
    A^alpha_nu = sum_{E - E' = nu} |E><E| A^alpha |E'><E'|, for every Bohr frequency
    nu in B(H) = {E_i - E_j} (the eigenvalue differences of `hamiltonian`).

    Returns (unique_nus, A_nu_all) with unique_nus shape (K,) and A_nu_all shape (n_ops, K, d, d).
    """
    deltaE = np.round(eigenvalues[:, None] - eigenvalues[None, :], nu_round)
    unique_nus = np.unique(deltaE)

    A_eig = eigenvectors.conj().T @ coupling_ops @ eigenvectors        # (n_ops, d, d)
    masks = (deltaE[None, :, :] == unique_nus[:, None, None]).astype(complex)  # (K, d, d)
    A_nu_all = np.einsum('ab,mkbc,dc->mkad', eigenvectors,
                          A_eig[:, None, :, :] * masks[None, :, :, :],
                          eigenvectors.conj())                          # (n_ops, K, d, d)
    return unique_nus, A_nu_all


def build_lindbladian_frequency_domain(hamiltonian: np.ndarray, beta: float, sigma: float, S: float,
                                        grid: IntegrationGrid, coupling_ops: np.ndarray,
                                        q_filter: str = "metropolis",
                                        eig_cache: Optional[Dict] = None,
                                        nu_round: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Bohr-frequency (Roberts QGS) construction. `grid` is accepted for a uniform call
    signature across all three constructions but is not used (no time integration here).
    """
    eigenvalues, eigenvectors = get_eig(hamiltonian, eig_cache)
    q_filter_func = Q_FILTER_FUNCTIONS[q_filter]
    d = hamiltonian.shape[0]

    unique_nus, A_nu_all = bohr_frequency_components(coupling_ops, eigenvalues, eigenvectors, nu_round)

    f_vec = filter_function_nu(unique_nus, beta, S, sigma, q_filter_func)  # (K,)

    nu_i, nu_j = unique_nus[:, None], unique_nus[None, :]
    weight_G = _prefactor_to_build_G(nu_i, nu_j, sigma, beta) * f_vec.conj()[:, None] * f_vec[None, :]
    weight_phi = _prefactor_for_phi(nu_i, nu_j, sigma) * f_vec.conj()[:, None] * f_vec[None, :]

    term = np.einsum('mvji,mwjl->mvwil', A_nu_all.conj(), A_nu_all)     # (n_ops,K,K,d,d)
    G_bohr = -np.einsum('vw,mvwil->il', weight_G, term)

    S4 = np.einsum('mvij,mwkl,vw->ikjl', A_nu_all.conj(), A_nu_all, weight_phi)
    phi_super = S4.reshape(d * d, d * d)

    U_super = superop_spre_plus_spostdag(G_bohr)
    Lindbladian = phi_super + U_super
    rho_gibbs = GibbsState(hamiltonian, beta).prepare()
    return Lindbladian, rho_gibbs


def construct_B_from_R_fast(hamiltonian: np.ndarray, R: np.ndarray, beta: float,
                             evals: np.ndarray, evecs: np.ndarray) -> np.ndarray:
    """B = sum_nu (i/2 * tanh(0.25*beta*nu)) * R_nu, built directly in the energy eigenbasis."""
    R_E = evecs.conj().T @ R @ evecs
    deltaE = evals[:, None] - evals[None, :]
    factor_mat = 0.5j * np.tanh(0.25 * beta * deltaE)
    B_E = factor_mat * R_E
    B = evecs @ B_E @ evecs.conj().T
    return 0.5 * (B + B.conj().T)  # enforce Hermiticity against numerical noise


def build_lindbladian_anthony(hamiltonian: np.ndarray, beta: float, sigma: float, S: float,
                               grid: IntegrationGrid, coupling_ops: np.ndarray,
                               q_filter: str = "metropolis",
                               eig_cache: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = get_eig(hamiltonian, eig_cache)
    q_filter_func = Q_FILTER_FUNCTIONS[q_filter]
    d = hamiltonian.shape[0]

    L_alpha = build_L_alpha(coupling_ops, eigenvalues, eigenvectors, grid, beta, S, sigma, q_filter_func)
    X_s = build_X_s(L_alpha, eigenvalues, eigenvectors, grid, sigma)
    kraus_ops = (X_s * np.sqrt(grid.dt)).reshape(-1, d, d)

    R = np.einsum('mij,mik->jk', kraus_ops.conj(), kraus_ops)   # sum_k C_k^dagger C_k
    B = construct_B_from_R_fast(hamiltonian, R, beta, eigenvalues, eigenvectors)

    Id = np.eye(d, dtype=complex)
    Lindbladian = (
        -1j * (np.kron(Id, B) - np.kron(B.T, Id))
        + kraus_ops_to_superoperator(kraus_ops)
        - 0.5 * np.kron(Id, R)
        - 0.5 * np.kron(R.T, Id)
    )
    rho_gibbs = GibbsState(hamiltonian, beta).prepare()
    return Lindbladian, rho_gibbs


def build_X_s_bohr(coupling_ops: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                    grid: IntegrationGrid, beta: float, S: float, sigma: float,
                    q_filter_func: Callable, nu_round: int = 10) -> np.ndarray:
    """X_s = sqrt(sigma*sqrt(2/pi)) * exp(-sigma^2 s^2) * sum_{nu in B(H)} f_hat(nu) * e^{i*s*nu}
    * A^alpha_nu, evaluated at every s in grid.t_grid -- the exact Bohr-frequency analog of
    build_X_s, using f_hat(nu) = filter_function_nu(nu, beta, S, sigma, q_filter_func) (the same
    frequency-domain filter used by build_lindbladian_frequency_domain) in place of the
    time-domain FFT used by build_L_alpha. The Gaussian window in s is still needed here: without
    it the s-integral used to build the Kraus operators below doesn't converge to a fixed,
    grid-independent value (X_s is otherwise purely oscillatory in s, never decaying).
    """
    unique_nus, A_nu_all = bohr_frequency_components(coupling_ops, eigenvalues, eigenvectors, nu_round)
    f_vec = filter_function_nu(unique_nus, beta, S, sigma, q_filter_func)  # (K,)

    phases = np.exp(1j * np.outer(grid.t_grid, unique_nus))  # (n_t, K)
    bohr_sum = np.einsum('tk,mkij->tmij', phases * f_vec[None, :], A_nu_all)  # (n_t, n_ops, d, d)
    window = prefactor_to_build_X(sigma, grid.t_grid)  # (n_t,)
    return bohr_sum * window[:, None, None, None]


def build_lindbladian_anthony_bohr(hamiltonian: np.ndarray, beta: float, sigma: float, S: float,
                                    grid: IntegrationGrid, coupling_ops: np.ndarray,
                                    q_filter: str = "metropolis",
                                    eig_cache: Optional[Dict] = None,
                                    nu_round: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Same as build_lindbladian_anthony, but X_s is built from the exact Bohr-frequency sum
    (build_X_s_bohr) instead of the time-domain FFT (build_L_alpha) + Gaussian time-window
    (build_X_s) pipeline.
    """
    eigenvalues, eigenvectors = get_eig(hamiltonian, eig_cache)
    q_filter_func = Q_FILTER_FUNCTIONS[q_filter]
    d = hamiltonian.shape[0]

    X_s = build_X_s_bohr(coupling_ops, eigenvalues, eigenvectors, grid, beta, S, sigma, q_filter_func,
                          nu_round=nu_round)
    kraus_ops = (X_s * np.sqrt(grid.dt)).reshape(-1, d, d)

    R = np.einsum('mij,mik->jk', kraus_ops.conj(), kraus_ops)   # sum_k C_k^dagger C_k
    B = construct_B_from_R_fast(hamiltonian, R, beta, eigenvalues, eigenvectors)

    Id = np.eye(d, dtype=complex)
    Lindbladian = (
        -1j * (np.kron(Id, B) - np.kron(B.T, Id))
        + kraus_ops_to_superoperator(kraus_ops)
        - 0.5 * np.kron(Id, R)
        - 0.5 * np.kron(R.T, Id)
    )
    rho_gibbs = GibbsState(hamiltonian, beta).prepare()
    return Lindbladian, rho_gibbs


CONSTRUCTION_METHODS: Dict[str, Callable] = {
    "time_domain": build_lindbladian_time_domain,
    "frequency_domain": build_lindbladian_frequency_domain,
    "anthony": build_lindbladian_anthony,
    "anthony_bohr": build_lindbladian_anthony_bohr,
}




def construct_Gamma_superoperator(rho_gibbs: np.ndarray) -> np.ndarray:
    """Superoperator matrix for rho -> sqrt(rho_gibbs) @ rho @ sqrt(rho_gibbs)."""
    evals, evecs = eigh(rho_gibbs)
    sqrt_rho = evecs @ np.diag(np.sqrt(np.clip(evals, 0.0, None))) @ evecs.conj().T
    return np.kron(sqrt_rho.conj(), sqrt_rho)


def l_dagger_identity_residual(Lindbladian: np.ndarray, dim: int) -> float:
    """max|L^dagger(I)|; should be ~0 (numerical noise only) for a valid generator."""
    I_vec = np.eye(dim, dtype=complex).reshape(-1, 1, order='F')
    L_dag_I = (Lindbladian.conj().T @ I_vec).reshape(dim, dim, order='F')
    return float(np.max(np.abs(L_dag_I)))


def detailed_balance_residual(Lindbladian: np.ndarray, rho_gibbs: np.ndarray) -> float:
    """Relative Frobenius-norm residual of G^{-1} L G = L^dagger; ~0 means detailed balance holds.

    (Uses Frobenius norm throughout for both the residual and the normalization -- a reasonable,
    consistently-applied analogue of the qutip Hilbert-Schmidt-distance check in the original
    notebook. What matters for this diagnostic is that it is near 0 when detailed balance holds
    and grows when it doesn't, which it does.)
    """
    G = construct_Gamma_superoperator(rho_gibbs)
    L_conjugated = Lindbladian @ G.conj().T
    L_dagger_term = G.conj().T @ Lindbladian.conj().T
    dist = np.linalg.norm(L_conjugated - L_dagger_term, 'fro')
    L_norm = np.linalg.norm(Lindbladian, 'fro')
    return float(dist / L_norm) if L_norm > 0 else float(dist)


def grid_with_sigma_and_beta_dependency(beta: float, sigma: float, base_t_max: float = 100.0, base_beta: float = 1.0,
                    base_n_t: int = 1201, min_points_per_sigma_window: int = 8) -> IntegrationGrid:
    """Grow the FFT time domain in proportion to beta, and shrink dt as sigma grows.

    t_max scales with beta as before: the KMS filter's frequency envelope sharpens as beta grows
    (~exp(-beta*|nu|/4)), so by the time-frequency uncertainty relation its time-domain
    counterpart decays more slowly and a fixed t_max increasingly truncates it. Scaling t_max ~
    beta (base_beta sets the beta at which t_max == base_t_max) keeps the truncation error
    roughly constant across a beta sweep.

    dt additionally shrinks with sigma: build_X_s / build_X_s_bohr multiply by a Gaussian window
    in time, prefactor_to_build_X(sigma, s) ~ exp(-sigma^2 s^2), whose width shrinks like 1/sigma.
    A beta-only dt ends up sampling that window with less than one grid point at large sigma --
    the discrete sum over grid.t_grid then badly undersamples the ds integral used to build the
    Kraus operators, which is why the anthony/anthony_bohr constructions' spectral gap spuriously
    blows up at large sigma while frequency_domain (which never discretizes over time) does not.

    Here dt is capped so at least `min_points_per_sigma_window` grid points fall within one
    window width (1/sigma), however large sigma gets, while never exceeding the base (beta-only)
    dt for small sigma where that isn't a concern.
    """
    base_dt = 2.0 * base_t_max / base_n_t
    t_max = base_t_max * max(beta, base_beta) / base_beta

    sigma_dt = 1.0 / (sigma * min_points_per_sigma_window)
    dt = min(base_dt, sigma_dt)
    return IntegrationGrid(t_max=t_max, dt=dt)


class evolve_until_converge:

    def __init__(self,
                 hamiltonian: np.ndarray,
                 _lindbladian,
                 rho_exact_gibbs: np.ndarray,
                 max_time: float,
                 threshold: float = 1e-3,
                 method: str = "SPECTRAL",
                 rho_init: Optional[np.ndarray] = None,
                 n_steps: Optional[int] = None,
                 bisection_iters: int = 60):

        self.hamiltonian = hamiltonian
        self._lindbladian = _lindbladian
        self.rho_exact_gibbs = rho_exact_gibbs
        self.max_time = max_time
        self.threshold = threshold
        self.method = method.upper()
        self.rho_init = rho_init
        self.n_steps = n_steps
        self.bisection_iters = bisection_iters

        # The Lindbladian may be handed in as a qutip superoperator (Qobj) or a plain matrix;
        # normalize once.
        self._lindbladian_matrix = (
            self._lindbladian.full() if hasattr(self._lindbladian, "full") else np.asarray(self._lindbladian)
        )

    def _apply_lindbladian(self, rho: np.ndarray) -> np.ndarray:
        rho_vec = rho.reshape(-1, 1, order="F")
        result_vec = self._lindbladian_matrix @ rho_vec
        return result_vec.reshape(rho.shape, order="F")

    def _default_rho_init(self, dim):
        rho_init = np.eye(dim) / dim
        return rho_init

    def _spectral_evolve(self, rho_init, rho_target, dim):
        L = self._lindbladian_matrix
        vec0 = rho_init.reshape(-1, 1, order="F")

        eigvals, W = np.linalg.eig(L)

        # Detailed balance -> spectrum should be real (up to numerical noise)
        eigvals = eigvals.real  # discard tiny imaginary parts from roundoff

        # Spectral gap = distance from 0 to the next-slowest decaying mode. Sort a
        # *copy* for this -- eigvals/W must stay index-aligned for the rho_at(t)
        # reconstruction below, which relies on eigvals[k] matching W[:, k].
        sorted_eigvals = np.sort(eigvals)[::-1]
        spectral_gap = -sorted_eigvals[1]  # sorted_eigvals[0] should be ~0 (steady state)


        cond = np.linalg.cond(W)

        if np.isfinite(cond) and cond < 1e10:
            Winv = np.linalg.inv(W)
            coeffs = Winv @ vec0

            def rho_at(t):
                vec_t = W @ (np.exp(eigvals * t)[:, None] * coeffs)
                return vec_t.reshape(dim, dim, order="F")
        else:
            # Fall back to direct matrix exponentials if the eigenbasis is ill-conditioned.
            def rho_at(t):
                vec_t = expm(L * t) @ vec0
                return vec_t.reshape(dim, dim, order="F")

        def td_at(t):
            return compute_trace_distance(rho_at(t), rho_target)

        td0 = td_at(0.0)
        if td0 <= self.threshold:
            return rho_init, 0.0, spectral_gap

        # Bracket search: grow t geometrically until trace distance drops below threshold.
        t_probe = max(self.max_time / 1.0e6, 1e-9)
        t_lo, t_hi = 0.0, None
        while t_probe <= self.max_time:
            if td_at(t_probe) <= self.threshold:
                t_hi = t_probe
                break
            t_lo = t_probe
            t_probe *= 2.0

        if t_hi is None:
            # Never converges within max_time.
            return rho_at(self.max_time), self.max_time, spectral_gap

        # Bisection refine.
        for _ in range(self.bisection_iters):
            t_mid = 0.5 * (t_lo + t_hi)
            if td_at(t_mid) <= self.threshold:
                t_hi = t_mid
            else:
                t_lo = t_mid

        return rho_at(t_hi), t_hi, spectral_gap

    def evolve_until_converge(self):
        if self.max_time <= 0:
            raise ValueError("max_time must be positive")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")

        dim = self.hamiltonian.shape[0]
        rho_init = self.rho_init if self.rho_init is not None else self._default_rho_init(dim)
        rho_curr = np.array(rho_init, dtype=complex)
        rho_target = np.array(self.rho_exact_gibbs, dtype=complex)

        if rho_curr.shape != (dim, dim):
            raise ValueError(f"rho_init must have shape {(dim, dim)}")
        if rho_target.shape != (dim, dim):
            raise ValueError(f"rho_exact_gibbs must have shape {(dim, dim)}")

        if self.method == "SPECTRAL":
            return self._spectral_evolve(rho_curr, rho_target, dim)

        n_steps = self.n_steps
        if n_steps is None:
            n_steps = max(100, int(75 * self.max_time))
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        dt = self.max_time / n_steps

        initial_td = compute_trace_distance(rho_curr, rho_target)
        if initial_td <= self.threshold:
            return rho_curr, 0.0

        if self.method == "RK4":
            for step in range(1, n_steps + 1):
                k1 = self._apply_lindbladian(rho_curr)
                k2 = self._apply_lindbladian(rho_curr + 0.5 * dt * k1)
                k3 = self._apply_lindbladian(rho_curr + 0.5 * dt * k2)
                k4 = self._apply_lindbladian(rho_curr + dt * k3)
                rho_curr = rho_curr + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                if compute_trace_distance(rho_curr, rho_target) <= self.threshold:
                    return rho_curr, step * dt
        elif self.method == "EULER":
            for step in range(1, n_steps + 1):
                rho_curr = rho_curr + dt * self._apply_lindbladian(rho_curr)
                if compute_trace_distance(rho_curr, rho_target) <= self.threshold:
                    return rho_curr, step * dt
        else:
            raise ValueError(f"Unknown integration method: {self.method}")

        return rho_curr, self.max_time


def run_convergence_sweep(hamiltonian: np.ndarray, sigma_values: Sequence[float], beta_values: Sequence[float],
                           grid_builder: Callable[[float, float], IntegrationGrid], S_value: float, coupling_ops: np.ndarray,
                           constructions: Sequence[str] = ("time_domain", "frequency_domain", "anthony", "anthony_bohr"),
                           q_filter: str = "metropolis", threshold: float = 1e-6, max_time: float = 1e6,
                           method: str = "SPECTRAL", rho_init: Optional[np.ndarray] = None,
                           l_dagger_tol: float = 1e-6, db_tol: float = 1e-6,
                           verbose: bool = True) -> pd.DataFrame:
    """Runs the sweep across every (construction, sigma, beta) combination and, for each,
    records the two structural diagnostics (L^dagger(I) residual, detailed-balance residual)
    so you can tell whether the convergence time at that point is measuring convergence to a
    valid physical steady state or an artifact of an invalid generator -- and whether trends
    are shared across constructions (more likely real physics) or specific to one
    (more likely a numerical artifact of that construction).

    `grid_builder(beta, sigma)` builds the FFT integration grid for a given (beta, sigma) pair --
    the grid is cached per (beta, sigma) pair. Pass `grid_with_sigma_and_beta_dependency`, which resolves the
    sigma-dependent Gaussian time window used by the time_domain/anthony/anthony_bohr
    constructions (see its docstring) in addition to the beta-dependent time window.
    """
    eig_cache: Dict = {}  # Hamiltonian is shared across all constructions and (sigma,beta) pairs
    grid_cache: Dict[Tuple[float, float], IntegrationGrid] = {}
    records = []
    combos = list(itertools.product(constructions, sigma_values, beta_values))
    dim = hamiltonian.shape[0]

    for i, (construction, sigma, beta) in enumerate(combos):
        if (beta, sigma) not in grid_cache:
            grid_cache[(beta, sigma)] = grid_builder(beta, sigma)
        grid = grid_cache[(beta, sigma)]

        builder = CONSTRUCTION_METHODS[construction]
        t0 = time.perf_counter()
        Lind, rho_g = builder(
            hamiltonian, beta, sigma, S_value, grid, coupling_ops, q_filter=q_filter, eig_cache=eig_cache)

        l_dag_res = l_dagger_identity_residual(Lind, dim)
        db_res = detailed_balance_residual(Lind, rho_g)
        valid_generator = (l_dag_res < l_dagger_tol) and (db_res < db_tol)

        evolver = evolve_until_converge(hamiltonian, Lind, rho_g, max_time, threshold, method, rho_init=rho_init)
        _, conv_time, spectral_gap = evolver.evolve_until_converge()
        wall = time.perf_counter() - t0
        converged = conv_time < max_time

        records.append({
            "construction": construction, "sigma": sigma, "beta": beta,
            "convergence_time": conv_time, "wall_time_sec": wall, "converged": converged,
            "l_dagger_id_residual": l_dag_res, "detailed_balance_residual": db_res,
            "valid_generator": valid_generator, "spectral_gap" : spectral_gap
        })

        if verbose:
            flags = []
            if not converged:
                flags.append("NOT CONVERGED")
            if not valid_generator:
                flags.append("INVALID GENERATOR")
            flag = ("  [" + ", ".join(flags) + "]") if flags else ""
            print(f"[{i+1}/{len(combos)}] {construction:17s} sigma={sigma:.4g} beta={beta:.4g} "
                  f"-> t_conv={conv_time:.6g}  L+_res={l_dag_res:.2e}  DB_res={db_res:.2e}"
                  f"  spectral_gap {spectral_gap:.6f}{flag} " )

    return pd.DataFrame(records)

CONSTRUCTION_ORDER = ["time_domain", "frequency_domain", "anthony", "anthony_bohr"]


def _heatmap(ax, df, value, log_scale=False, cmap=None, title="", cbar_label=""):
    pivot = df.pivot(index="beta", columns="sigma", values=value)
    data = pivot.values.astype(float)
    if log_scale:
        with np.errstate(invalid="ignore", divide="ignore"):
            data = np.log10(np.clip(data, 1e-300, None))
    im = ax.imshow(
        data, aspect="auto", origin="lower", cmap=cmap,
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
    )
    ax.set_xlabel("sigma")
    ax.set_ylabel("beta")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    return pivot


def plot_convergence_heatmap(df: pd.DataFrame, log_scale: bool = True,
                              constructions: Sequence[str] = CONSTRUCTION_ORDER):
    """One convergence-time heatmap per construction, side by side, on a shared color scale
    """
    present = [c for c in constructions if c in set(df["construction"])]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5), squeeze=False)
    for ax, construction in zip(axes[0], present):
        sub = df[df["construction"] == construction]
        _heatmap(ax, sub, "convergence_time", log_scale=log_scale, title=construction,
                  cbar_label="log10(convergence time)" if log_scale else "convergence time")
    fig.suptitle("Convergence time vs sigma & beta, by construction")
    plt.tight_layout()
    plt.show()


def plot_convergence_lines(df: pd.DataFrame, constructions: Sequence[str] = CONSTRUCTION_ORDER):
    present = [c for c in constructions if c in set(df["construction"])]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5), squeeze=False, sharey=True)
    for ax, construction in zip(axes[0], present):
        sub = df[df["construction"] == construction]
        for beta, group in sub.groupby("beta"):
            group = group.sort_values("sigma")
            ax.plot(group["sigma"], group["convergence_time"], marker="o", label=f"beta={beta:.3g}")
        ax.set_xlabel("sigma")
        ax.set_ylabel("convergence time")
        ax.set_yscale("log")
        ax.set_title(construction)
        ax.legend(fontsize=8)
    fig.suptitle("Convergence time vs sigma, for each beta, by construction")
    plt.tight_layout()
    plt.show()


def plot_spectral_gap_lines(df: pd.DataFrame, constructions: Sequence[str] = CONSTRUCTION_ORDER):
    present = [c for c in constructions if c in set(df["construction"])]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5), squeeze=False, sharey=True)
    for ax, construction in zip(axes[0], present):
        sub = df[df["construction"] == construction]
        for beta, group in sub.groupby("beta"):
            group = group.sort_values("sigma")
            ax.plot(group["sigma"], group["spectral_gap"], marker="o", label=f"beta={beta:.3g}")
        ax.set_xlabel("sigma")
        ax.set_ylabel("spectral gap")
        ax.set_yscale("log")
        ax.set_title(construction)
        ax.legend(fontsize=8)
    fig.suptitle("Spectral gap vs sigma, for each beta, by construction")
    plt.tight_layout()
    plt.show()


def hastle_function_inverse_sigma(sigma, beta):
    return 1.0 / sigma
 
 
def hastle_function_inverse_sigma_beta(sigma, beta):
    return 1.0 / (sigma * beta)
 
 
def hastle_function_inverse_sigma_sq(sigma, beta):
    return 1.0 / (sigma ** 2)
 

def hastle_function_inverse_sigma_beta_sq(sigma, beta):
    return 1.0 / (sigma * beta**2)

def hastle_function_constant(sigma, beta):
    return np.ones_like(np.asarray(sigma, dtype=float))
 
 
HASTLE_FUNCTIONS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "inverse_sigma": hastle_function_inverse_sigma,
    "inverse_sigma_beta": hastle_function_inverse_sigma_beta,
    "inverse_sigma_sq": hastle_function_inverse_sigma_sq,
    "inverse_sigma_beta_sq": hastle_function_inverse_sigma_beta_sq,
    "constant": hastle_function_constant,
}
 
 
def convergence_time_times_hastle(
    df: pd.DataFrame,
    constructions: Sequence[str] = CONSTRUCTION_ORDER,
    hastle_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = hastle_function_inverse_sigma,
) -> Dict[str, pd.DataFrame]:
    """Builds, for each construction, a (sigma x beta) table of convergence_time * hastle_function.
 
    `hastle_function` is any callable f(sigma, beta) -> weight, applied elementwise over the
    (sigma, beta) grid. Pass one of the functions in `HASTLE_FUNCTIONS`, or your own callable
    with the same signature, from the run notebook.
    """
    weighted_by_construction = {}
    for construction, sub in df.groupby('construction'):
        pivot = sub.pivot_table(index='sigma', columns='beta', values='convergence_time')
        S, B = np.meshgrid(pivot.index.values, pivot.columns.values, indexing='ij')
        f_grid = pd.DataFrame(hastle_function(S, B), index=pivot.index, columns=pivot.columns)
        weighted_by_construction[construction] = pivot * f_grid
 
    return weighted_by_construction
 
 
def plot_convergence_time_times_hastle(weighted_by_construction: Dict[str, pd.DataFrame],
                         constructions: Sequence[str] = CONSTRUCTION_ORDER,
                         hastle_name: Optional[str] = None):
    """weighted vs sigma, one line per beta, one subplot per construction.
 
    `hastle_name`, if given, is shown in the figure title so it's clear which weighting
    was used to produce the plot.
    """
    present = [c for c in constructions if c in weighted_by_construction]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5), squeeze=False, sharey=True)
    for ax, construction in zip(axes[0], present):
        weighted = weighted_by_construction[construction]
        for beta in weighted.columns:
            ax.plot(weighted.index, weighted[beta], marker="o", label=f"beta={beta:.3g}")
        ax.set_xlabel("sigma")
        ax.set_ylabel("weighted convergence time")
        ax.set_title(construction)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    title = "Convergence time times HASTLE w.r.t. sigma, for each beta, by construction"
    if hastle_name is not None:
        title += f"  (weighting: {hastle_name})"
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def find_sigma_minima(weighted_by_construction: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For each (construction, beta), locate the sigma at which the weighted convergence time
    (as returned by `convergence_time_times_hastle`) is smallest.

    A minimum only counts as an interior minimum if it falls strictly between two sampled sigma
    points -- an argmin sitting on either edge of the grid usually means the curve is monotonic
    there, or the true minimum lies outside the sampled range, so `has_interior_minimum` is False
    in that case (and `sigma_lo`/`sigma_hi` -- the bracket to refine within -- are left as None).
    """
    rows = []
    for construction, weighted in weighted_by_construction.items():
        sigmas = weighted.index.values
        for beta in weighted.columns:
            values = weighted[beta].values
            k = int(np.nanargmin(values))
            is_interior = 0 < k < len(sigmas) - 1
            rows.append({
                "construction": construction, "beta": beta,
                "min_sigma": sigmas[k], "min_value": values[k],
                "has_interior_minimum": is_interior,
                "sigma_lo": sigmas[k - 1] if is_interior else None,
                "sigma_hi": sigmas[k + 1] if is_interior else None,
            })
    return pd.DataFrame(rows)


def refine_sigma_minima(
        hamiltonian: np.ndarray, minima_df: pd.DataFrame,
        grid_builder: Callable[[float, float], IntegrationGrid], S_value: float, coupling_ops: np.ndarray,
        n_sigma_fine: int = 25,
        hastle_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = hastle_function_inverse_sigma,
        q_filter: str = "metropolis", threshold: float = 1e-6, max_time: float = 1e6,
        method: str = "SPECTRAL", rho_init: Optional[np.ndarray] = None,
        verbose: bool = False) -> Tuple[pd.DataFrame, Dict[Tuple[str, float], pd.DataFrame]]:
    """For every row of `minima_df` that has an interior minimum, re-sweep sigma on a finer grid
    spanning the two coarse grid points bracketing that minimum, and locate the minimum again
    within that narrower window -- giving a more precise (sigma, weighted value) estimate than
    the coarse grid alone can resolve.

    `grid_builder(beta, sigma)` -- see `run_convergence_sweep`.

    Returns (refined_minima_df, fine_results_by_key), where fine_results_by_key maps
    (construction, beta) -> the raw fine-grid sweep DataFrame, for further inspection or plotting.
    """
    refined_rows = []
    fine_results_by_key = {}

    for _, row in minima_df.iterrows():
        if not row["has_interior_minimum"]:
            continue
        construction, beta = row["construction"], row["beta"]
        fine_sigmas = np.linspace(row["sigma_lo"], row["sigma_hi"], n_sigma_fine)

        fine_df = run_convergence_sweep(
            hamiltonian, fine_sigmas, [beta], grid_builder, S_value, coupling_ops,
            constructions=(construction,), q_filter=q_filter, threshold=threshold,
            max_time=max_time, method=method, rho_init=rho_init, verbose=verbose)
        fine_results_by_key[(construction, beta)] = fine_df

        fine_weighted = convergence_time_times_hastle(
            fine_df, constructions=(construction,), hastle_function=hastle_function)
        pivot = fine_weighted[construction]
        values = pivot[beta].values
        k = int(np.nanargmin(values))

        refined_rows.append({
            "construction": construction, "beta": beta,
            "coarse_min_sigma": row["min_sigma"], "coarse_min_value": row["min_value"],
            "refined_min_sigma": pivot.index.values[k], "refined_min_value": values[k],
        })

    return pd.DataFrame(refined_rows), fine_results_by_key


def sweep_and_locate_sigma_minima(
        hamiltonian: np.ndarray, sigma_values: Sequence[float], beta_values: Sequence[float],
        grid_builder: Callable[[float, float], IntegrationGrid], S_value: float, coupling_ops: np.ndarray,
        constructions: Sequence[str] = CONSTRUCTION_ORDER,
        hastle_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = hastle_function_inverse_sigma,
        n_sigma_fine: int = 25,
        q_filter: str = "metropolis", threshold: float = 1e-6, max_time: float = 1e6,
        method: str = "SPECTRAL", rho_init: Optional[np.ndarray] = None,
        verbose: bool = True) -> Dict:
    """End-to-end: a dense coarse sigma sweep -> weighted convergence time -> per-(construction,
    beta) sigma minimum -> refined minimum on a finer local grid around each one found.

    This is the same computation as `run_convergence_sweep` + `convergence_time_times_hastle`,
    just with (typically) many more sigma points, plus the minimum-finding and refinement on top.

    `grid_builder(beta, sigma)` -- see `run_convergence_sweep`.

    Returns a dict with keys:
      "coarse_results_df"  -- raw sweep output on `sigma_values`
      "coarse_weighted"    -- {construction: pivot(sigma, beta)}, same shape as
                               `convergence_time_times_hastle`'s return value
      "minima_df"          -- one row per (construction, beta) locating the coarse-grid minimum
      "refined_minima_df"  -- one row per (construction, beta) that had an interior minimum, with
                               the finer-grid refined sigma/value
      "refined_results"    -- {(construction, beta): fine-grid sweep_df}
    """
    coarse_results_df = run_convergence_sweep(
        hamiltonian, sigma_values, beta_values, grid_builder, S_value, coupling_ops,
        constructions=constructions, q_filter=q_filter, threshold=threshold,
        max_time=max_time, method=method, rho_init=rho_init, verbose=verbose)

    coarse_weighted = convergence_time_times_hastle(
        coarse_results_df, constructions=constructions, hastle_function=hastle_function)

    minima_df = find_sigma_minima(coarse_weighted)

    refined_minima_df, refined_results = refine_sigma_minima(
        hamiltonian, minima_df, grid_builder, S_value, coupling_ops,
        n_sigma_fine=n_sigma_fine, hastle_function=hastle_function,
        q_filter=q_filter, threshold=threshold, max_time=max_time, method=method,
        rho_init=rho_init, verbose=verbose)

    if verbose:
        n_found = int(minima_df["has_interior_minimum"].sum())
        print(f"\nInterior sigma minimum found for {n_found} / {len(minima_df)} "
              f"(construction, beta) combinations; refined on a {n_sigma_fine}-point local grid.")
        missing = minima_df[~minima_df["has_interior_minimum"]]
        if len(missing):
            print("No interior minimum (argmin sits on the sigma grid edge) for:")
            for _, row in missing.iterrows():
                print(f"  {row['construction']:17s} beta={row['beta']:.4g}  "
                      f"(argmin at sigma={row['min_sigma']:.4g})")

    return {
        "coarse_results_df": coarse_results_df,
        "coarse_weighted": coarse_weighted,
        "minima_df": minima_df,
        "refined_minima_df": refined_minima_df,
        "refined_results": refined_results,
    }


def plot_sigma_minima_vs_beta(minima_df: pd.DataFrame,
                               constructions: Sequence[str] = CONSTRUCTION_ORDER,
                               value_col: str = "refined_min_sigma",
                               ylabel: Optional[str] = None) -> None:
    """Plot the located sigma minimum against beta, one line per construction.

    Intended for `sweep_and_locate_sigma_minima`'s "refined_minima_df" (default `value_col`
    "refined_min_sigma", the sigma location of the minimum; pass "refined_min_value" to plot the
    minimum weighted convergence time itself instead). Also accepts the coarse "minima_df" if you
    pass `value_col="min_sigma"` or `"min_value"`.

    Only betas where an interior minimum was actually found are plotted -- if `minima_df` still
    has a "has_interior_minimum" column (true of the coarse "minima_df"; "refined_minima_df"
    already omits these rows), rows where it's False are dropped first.
    """
    if "has_interior_minimum" in minima_df.columns:
        minima_df = minima_df[minima_df["has_interior_minimum"]]

    present = [c for c in constructions if c in set(minima_df["construction"])]
    fig, ax = plt.subplots(figsize=(7, 5))
    for construction in present:
        sub = minima_df[minima_df["construction"] == construction].sort_values("beta")
        ax.plot(sub["beta"], sub[value_col], marker="o", label=construction)
    ax.set_xlabel("beta")
    ax.set_ylabel(ylabel or value_col.replace("_", " "))
    ax.set_title("Sigma minimum vs beta, by construction (only where an interior minimum was found)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()