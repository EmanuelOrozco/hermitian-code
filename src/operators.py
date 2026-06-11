"""Operadores discretos: derivadas, rotacional, A, B y Ĥ."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import sparse

from .constants import (
    MU0,
    Polarization,
    TEBlocks,
    TMBlocks,
    block_offset,
    flat_index,
)
from .grid import YeeGrid, bloch_phase_x, bloch_phase_y


def build_Df_x(grid: YeeGrid, kx: float) -> sparse.csr_matrix:
    """Diferencia finita hacia adelante en x con BC de Bloch."""
    n = grid.N
    n2 = grid.n_dof
    dxinv = 1.0 / grid.delta
    phase_p = bloch_phase_x(kx)
    rows, cols, data = [], [], []
    for j in range(n):
        for i in range(n):
            row = flat_index(i, j, n)
            rows.append(row)
            cols.append(row)
            data.append(-dxinv)
            if i + 1 < n:
                rows.append(row)
                cols.append(flat_index(i + 1, j, n))
                data.append(dxinv)
            else:
                rows.append(row)
                cols.append(flat_index(0, j, n))
                data.append(phase_p * dxinv)
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(n2, n2), dtype=np.complex128
    )


def build_Df_y(grid: YeeGrid, ky: float) -> sparse.csr_matrix:
    """Diferencia finita hacia adelante en y con BC de Bloch."""
    n = grid.N
    n2 = grid.n_dof
    dyinv = 1.0 / grid.delta
    phase_p = bloch_phase_y(ky)
    rows, cols, data = [], [], []
    for j in range(n):
        for i in range(n):
            row = flat_index(i, j, n)
            rows.append(row)
            cols.append(row)
            data.append(-dyinv)
            if j + 1 < n:
                rows.append(row)
                cols.append(flat_index(i, j + 1, n))
                data.append(dyinv)
            else:
                rows.append(row)
                cols.append(flat_index(i, 0, n))
                data.append(phase_p * dyinv)
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(n2, n2), dtype=np.complex128
    )


def build_Db_x(grid: YeeGrid, kx: float) -> sparse.csr_matrix:
    """Diferencia finita hacia atrás en x (= −Df_x† con fase conjugada)."""
    n = grid.N
    n2 = grid.n_dof
    dxinv = 1.0 / grid.delta
    phase_m = np.conj(bloch_phase_x(kx))
    rows, cols, data = [], [], []
    for j in range(n):
        for i in range(n):
            row = flat_index(i, j, n)
            rows.append(row)
            cols.append(row)
            data.append(dxinv)
            if i - 1 >= 0:
                rows.append(row)
                cols.append(flat_index(i - 1, j, n))
                data.append(-dxinv)
            else:
                rows.append(row)
                cols.append(flat_index(n - 1, j, n))
                data.append(phase_m * (-dxinv))
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(n2, n2), dtype=np.complex128
    )


def build_Db_y(grid: YeeGrid, ky: float) -> sparse.csr_matrix:
    """Diferencia finita hacia atrás en y."""
    n = grid.N
    n2 = grid.n_dof
    dyinv = 1.0 / grid.delta
    phase_m = np.conj(bloch_phase_y(ky))
    rows, cols, data = [], [], []
    for j in range(n):
        for i in range(n):
            row = flat_index(i, j, n)
            rows.append(row)
            cols.append(row)
            data.append(dyinv)
            if j - 1 >= 0:
                rows.append(row)
                cols.append(flat_index(i, j - 1, n))
                data.append(-dyinv)
            else:
                rows.append(row)
                cols.append(flat_index(i, n - 1, n))
                data.append(phase_m * (-dyinv))
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(n2, n2), dtype=np.complex128
    )


def build_CE_TM(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """C_E^TM: (2N²)×(N²), mapea Ez → (Hx, Hy)."""
    n2 = grid.n_dof
    dfy = build_Df_y(grid, ky)
    dfx = build_Df_x(grid, kx)
    upper = sparse.vstack([dfy, -dfx], format="csr")
    return upper


def build_CH_TM(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """C_H^TM = (C_E^TM)†."""
    return build_CE_TM(grid, kx, ky).conj().transpose().tocsr()


def build_CE_TE(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """C_E^TE: (N²)×(2N²), mapea (Ex, Ey) → Hz."""
    n2 = grid.n_dof
    dfy = build_Df_y(grid, ky)
    dfx = build_Df_x(grid, kx)
    left = -dfy
    right = dfx
    return sparse.hstack([left, right], format="csr")


def build_CH_TE(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """C_H^TE = (C_E^TE)†."""
    return build_CE_TE(grid, kx, ky).conj().transpose().tocsr()


def build_I_metal(grid: YeeGrid) -> sparse.csr_matrix:
    """Identidad restringida a nodos metálicos."""
    n2 = grid.n_dof
    idx = np.where(grid.metal_mask)[0]
    data = np.ones(len(idx), dtype=np.complex128)
    return sparse.csr_matrix(
        (data, (idx, idx)), shape=(n2, n2), dtype=np.complex128
    )


def build_A_diagonal_TM(grid: YeeGrid) -> np.ndarray:
    """Diagonal de A para TM-Drude."""
    n2 = grid.n_dof
    a = np.zeros(TMBlocks.N_BLOCKS_DRUDE * n2, dtype=np.float64)
    for k in range(n2):
        a[TMBlocks.BLK_HX * n2 + k] = MU0
        a[TMBlocks.BLK_HY * n2 + k] = MU0
        a[TMBlocks.BLK_EZ * n2 + k] = grid.eps_z[k]
        eps_i = grid.eps_inf_mask[k]
        wp_i = grid.omega_p_mask[k]
        if wp_i > 0:
            a[TMBlocks.BLK_VZ * n2 + k] = 1.0 / (eps_i * wp_i**2)
        else:
            a[TMBlocks.BLK_VZ * n2 + k] = 1.0
    return a


def build_A_diagonal_TE(grid: YeeGrid) -> np.ndarray:
    """Diagonal de A para TE-Drude."""
    n2 = grid.n_dof
    a = np.zeros(TEBlocks.N_BLOCKS_DRUDE * n2, dtype=np.float64)
    for k in range(n2):
        a[TEBlocks.BLK_HZ * n2 + k] = MU0
        a[TEBlocks.BLK_EX * n2 + k] = grid.eps_x[k]
        a[TEBlocks.BLK_EY * n2 + k] = grid.eps_y[k]
        eps_x = grid.eps_x[k]
        eps_y = grid.eps_y[k]
        wp_k = grid.omega_p_mask[k]
        if wp_k > 0:
            a[TEBlocks.BLK_VX * n2 + k] = 1.0 / (eps_x * wp_k**2)
            a[TEBlocks.BLK_VY * n2 + k] = 1.0 / (eps_y * wp_k**2)
        else:
            a[TEBlocks.BLK_VX * n2 + k] = 1.0
            a[TEBlocks.BLK_VY * n2 + k] = 1.0
    return a


def build_A_diagonal(grid: YeeGrid) -> np.ndarray:
    if grid.mode == Polarization.TM:
        return build_A_diagonal_TM(grid)
    return build_A_diagonal_TE(grid)


def build_sqrtA_inv(grid: YeeGrid) -> np.ndarray:
    """Vector s_inv = 1/√A_kk."""
    a = build_A_diagonal(grid)
    return np.where(a > 0, 1.0 / np.sqrt(a), 0.0)


def build_B_TM(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """Matriz dinámica B (Hermitiana) para TM-Drude."""
    n2 = grid.n_dof
    ntot = TMBlocks.N_BLOCKS_DRUDE * n2
    ce = build_CE_TM(grid, kx, ky)
    ch = build_CH_TM(grid, kx, ky)
    im = build_I_metal(grid)

    blocks = [
        [None, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
    ]
    blocks[0][2] = (-1j * ce)[:n2, :]
    blocks[1][2] = (-1j * ce)[n2:, :]
    blocks[2][0] = 1j * ch[:, :n2]
    blocks[2][1] = 1j * ch[:, n2:]
    blocks[2][3] = -1j * im
    blocks[3][2] = 1j * im

    return sparse.bmat(blocks, format="csr", dtype=np.complex128)


def build_B_TE(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """Matriz dinámica B (Hermitiana) para TE-Drude."""
    n2 = grid.n_dof
    ce = build_CE_TE(grid, kx, ky)
    ch = build_CH_TE(grid, kx, ky)
    im = build_I_metal(grid)

    blocks = [[None] * 5 for _ in range(5)]
    blocks[0][1] = -1j * ce[:, :n2]
    blocks[0][2] = -1j * ce[:, n2:]
    # (E, Hz): +i CH — adjunto de (Hz, E) = −i CE  (misma convención que TM)
    blocks[1][0] = 1j * ch[:n2, :]
    blocks[2][0] = 1j * ch[n2:, :]
    blocks[1][3] = -1j * im
    blocks[2][4] = -1j * im
    blocks[3][1] = 1j * im
    blocks[4][2] = 1j * im

    return sparse.bmat(blocks, format="csr", dtype=np.complex128)


def build_B(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    if grid.mode == Polarization.TM:
        return build_B_TM(grid, kx, ky)
    return build_B_TE(grid, kx, ky)


def apply_similarity(
    b: sparse.csr_matrix, s_inv: np.ndarray
) -> sparse.csr_matrix:
    """Ĥ_ij = s_inv[i] B_ij s_inv[j] (eq. 12)."""
    d = sparse.diags(s_inv.astype(np.complex128))
    return (d @ b @ d).tocsr()


def build_H_hat(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """Hamiltoniano Hermitiano Ĥ = A^{-1/2} B A^{-1/2}."""
    b = build_B(grid, kx, ky)
    s_inv = build_sqrtA_inv(grid)
    return apply_similarity(b, s_inv)


def build_loss_perturbation(
    grid: YeeGrid, gamma: float
) -> sparse.csr_matrix:
    """δĤ anti-Hermitiana en bloques V (solo metal)."""
    n2 = grid.n_dof
    ntot = grid.total_dof
    rows, cols, data = [], [], []

    if grid.mode == Polarization.TM:
        v_blk = TMBlocks.BLK_VZ
        for k in range(n2):
            if not grid.metal_mask[k]:
                continue
            eps_i = grid.eps_inf_mask[k]
            wp_i = grid.omega_p_mask[k]
            if wp_i <= 0:
                continue
            idx = block_offset(v_blk, n2) + k
            val = -1j * gamma / (eps_i * wp_i**2)
            rows.append(idx)
            cols.append(idx)
            data.append(val)
    else:
        for k in range(n2):
            if not grid.metal_mask[k]:
                continue
            wp = grid.omega_p_mask[k]
            if wp <= 0:
                continue
            for v_blk, eps_k in (
                (TEBlocks.BLK_VX, grid.eps_x[k]),
                (TEBlocks.BLK_VY, grid.eps_y[k]),
            ):
                idx = block_offset(v_blk, n2) + k
                val = -1j * gamma / (eps_k * wp**2)
                rows.append(idx)
                cols.append(idx)
                data.append(val)

    if not rows:
        return sparse.csr_matrix((ntot, ntot), dtype=np.complex128)
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(ntot, ntot), dtype=np.complex128
    )


def assemble_systems(
    grid: YeeGrid, kx: float, ky: float, gamma: float = 0.0
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Retorna (Ĥ₀, Ĥ_lossy, A_diagonal)."""
    h0 = build_H_hat(grid, kx, ky)
    a_diag = build_A_diagonal(grid)
    if gamma > 0:
        delta = build_loss_perturbation(grid, gamma)
        h_lossy = (h0 + delta).tocsr()
    else:
        h_lossy = h0
    return h0, h_lossy, a_diag
