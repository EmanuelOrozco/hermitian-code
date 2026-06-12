"""Operadores discretos: derivadas, rotacional, A, B y Ĥ.

Implementación Drude reducida del formalismo de Raman & Fan.

Lossless:
    ω A x = B x
    y = sqrt(A) x
    ω y = A^{-1/2} B A^{-1/2} y

Para Drude, ω0 = 0 y el bloque P no aparece.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import sparse

from .constants import MU0, Polarization, TEBlocks, TMBlocks, block_offset, flat_index
from .grid import YeeGrid, bloch_phase_x, bloch_phase_y


def build_Df_x(grid: YeeGrid, kx: float) -> sparse.csr_matrix:
    """Diferencia finita hacia adelante en x con BC de Bloch."""
    n = grid.N
    n2 = grid.n_dof
    dxinv = 1.0 / grid.delta
    phase_p = bloch_phase_x(kx)

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    for i in range(n):
        for j in range(n):
            row = flat_index(i, j, n)

            rows.append(row)
            cols.append(row)
            data.append(-dxinv)

            if i + 1 < n:
                col = flat_index(i + 1, j, n)
                val = dxinv
            else:
                col = flat_index(0, j, n)
                val = phase_p * dxinv

            rows.append(row)
            cols.append(col)
            data.append(val)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n2, n2), dtype=np.complex128)


def build_Df_y(grid: YeeGrid, ky: float) -> sparse.csr_matrix:
    """Diferencia finita hacia adelante en y con BC de Bloch."""
    n = grid.N
    n2 = grid.n_dof
    dyinv = 1.0 / grid.delta
    phase_p = bloch_phase_y(ky)

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    for i in range(n):
        for j in range(n):
            row = flat_index(i, j, n)

            rows.append(row)
            cols.append(row)
            data.append(-dyinv)

            if j + 1 < n:
                col = flat_index(i, j + 1, n)
                val = dyinv
            else:
                col = flat_index(i, 0, n)
                val = phase_p * dyinv

            rows.append(row)
            cols.append(col)
            data.append(val)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n2, n2), dtype=np.complex128)


def build_Db_x(grid: YeeGrid, kx: float) -> sparse.csr_matrix:
    """Diferencia hacia atrás compatible: Db_x = -Df_x†."""
    return -build_Df_x(grid, kx).conj().transpose().tocsr()


def build_Db_y(grid: YeeGrid, ky: float) -> sparse.csr_matrix:
    """Diferencia hacia atrás compatible: Db_y = -Df_y†."""
    return -build_Df_y(grid, ky).conj().transpose().tocsr()


def build_CE_TM(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """C_E^TM: Ez -> (Hx, Hy)."""
    dfy = build_Df_y(grid, ky)
    dfx = build_Df_x(grid, kx)
    return sparse.vstack([dfy, -dfx], format="csr")


def build_CH_TM(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """C_H^TM = C_E^TM†."""
    return build_CE_TM(grid, kx, ky).conj().transpose().tocsr()


def build_CE_TE(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """C_E^TE: (Ex, Ey) -> Hz."""
    dfy = build_Df_y(grid, ky)
    dfx = build_Df_x(grid, kx)
    return sparse.hstack([-dfy, dfx], format="csr")


def build_CH_TE(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """C_H^TE = C_E^TE†."""
    return build_CE_TE(grid, kx, ky).conj().transpose().tocsr()


def _diag_from_mask(mask: np.ndarray) -> sparse.csr_matrix:
    n2 = mask.size
    idx = np.where(mask)[0]
    data = np.ones(len(idx), dtype=np.complex128)
    return sparse.csr_matrix((data, (idx, idx)), shape=(n2, n2), dtype=np.complex128)


def build_I_metal(grid: YeeGrid) -> sparse.csr_matrix:
    """Identidad restringida a nodos metálicos."""
    return _diag_from_mask(grid.metal_mask)


def build_I_metal_x(grid: YeeGrid) -> sparse.csr_matrix:
    """Identidad restringida a aristas Ex metálicas."""
    return _diag_from_mask(grid.metal_x)


def build_I_metal_y(grid: YeeGrid) -> sparse.csr_matrix:
    """Identidad restringida a aristas Ey metálicas."""
    return _diag_from_mask(grid.metal_y)


def build_A_diagonal_TM(grid: YeeGrid) -> np.ndarray:
    """Diagonal de A para TM-Drude: x = (Hx, Hy, Ez, Vz)."""
    n2 = grid.n_dof
    a = np.ones(TMBlocks.N_BLOCKS_DRUDE * n2, dtype=np.float64)

    a[block_offset(TMBlocks.BLK_HX, n2) : block_offset(TMBlocks.BLK_HX, n2) + n2] = MU0
    a[block_offset(TMBlocks.BLK_HY, n2) : block_offset(TMBlocks.BLK_HY, n2) + n2] = MU0
    a[block_offset(TMBlocks.BLK_EZ, n2) : block_offset(TMBlocks.BLK_EZ, n2) + n2] = (
        grid.eps_z
    )

    v0 = block_offset(TMBlocks.BLK_VZ, n2)
    for k in range(n2):
        wp = grid.omega_p_mask[k]
        eps = grid.eps_inf_mask[k]
        if grid.metal_mask[k] and wp > 0:
            a[v0 + k] = 1.0 / (eps * wp**2)
        else:
            # A positivo y acoplamiento B=0 fuera del metal.
            a[v0 + k] = 1.0

    return a


def build_A_diagonal_TE(grid: YeeGrid) -> np.ndarray:
    """Diagonal de A para TE-Drude: x = (Hz, Ex, Ey, Vx, Vy)."""
    n2 = grid.n_dof
    a = np.ones(TEBlocks.N_BLOCKS_DRUDE * n2, dtype=np.float64)

    a[block_offset(TEBlocks.BLK_HZ, n2) : block_offset(TEBlocks.BLK_HZ, n2) + n2] = MU0
    a[block_offset(TEBlocks.BLK_EX, n2) : block_offset(TEBlocks.BLK_EX, n2) + n2] = (
        grid.eps_x
    )
    a[block_offset(TEBlocks.BLK_EY, n2) : block_offset(TEBlocks.BLK_EY, n2) + n2] = (
        grid.eps_y
    )

    vx0 = block_offset(TEBlocks.BLK_VX, n2)
    vy0 = block_offset(TEBlocks.BLK_VY, n2)

    for k in range(n2):
        wp_x = grid.omega_p_x[k]
        wp_y = grid.omega_p_y[k]

        if grid.metal_x[k] and wp_x > 0:
            a[vx0 + k] = 1.0 / (grid.eps_x[k] * wp_x**2)
        else:
            a[vx0 + k] = 1.0

        if grid.metal_y[k] and wp_y > 0:
            a[vy0 + k] = 1.0 / (grid.eps_y[k] * wp_y**2)
        else:
            a[vy0 + k] = 1.0

    return a


def build_A_diagonal(grid: YeeGrid) -> np.ndarray:
    if grid.mode == Polarization.TM:
        return build_A_diagonal_TM(grid)
    return build_A_diagonal_TE(grid)


def build_sqrtA_inv(grid: YeeGrid) -> np.ndarray:
    a = build_A_diagonal(grid)
    return np.where(a > 0, 1.0 / np.sqrt(a), 0.0)


def build_B_TM(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """Matriz B Hermitiana para TM-Drude."""
    n2 = grid.n_dof
    ce = build_CE_TM(grid, kx, ky)
    ch = build_CH_TM(grid, kx, ky)
    im = build_I_metal(grid)

    blocks = [[None] * 4 for _ in range(4)]

    blocks[0][2] = (-1j * ce)[:n2, :]
    blocks[1][2] = (-1j * ce)[n2:, :]

    blocks[2][0] = 1j * ch[:, :n2]
    blocks[2][1] = 1j * ch[:, n2:]
    blocks[2][3] = -1j * im

    blocks[3][2] = 1j * im

    return sparse.bmat(blocks, format="csr", dtype=np.complex128)


def build_B_TE(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    """Matriz B Hermitiana para TE-Drude."""
    n2 = grid.n_dof
    ce = build_CE_TE(grid, kx, ky)
    ch = build_CH_TE(grid, kx, ky)

    imx = build_I_metal_x(grid)
    imy = build_I_metal_y(grid)

    blocks = [[None] * 5 for _ in range(5)]

    blocks[0][1] = -1j * ce[:, :n2]
    blocks[0][2] = -1j * ce[:, n2:]

    blocks[1][0] = 1j * ch[:n2, :]
    blocks[2][0] = 1j * ch[n2:, :]

    blocks[1][3] = -1j * imx
    blocks[2][4] = -1j * imy

    blocks[3][1] = 1j * imx
    blocks[4][2] = 1j * imy

    return sparse.bmat(blocks, format="csr", dtype=np.complex128)


def build_B(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    if grid.mode == Polarization.TM:
        return build_B_TM(grid, kx, ky)
    return build_B_TE(grid, kx, ky)


def apply_similarity(b: sparse.csr_matrix, s_inv: np.ndarray) -> sparse.csr_matrix:
    """A^{-1/2} B A^{-1/2}."""
    d = sparse.diags(s_inv.astype(np.complex128))
    return (d @ b @ d).tocsr()


def build_H_hat(grid: YeeGrid, kx: float, ky: float) -> sparse.csr_matrix:
    b = build_B(grid, kx, ky)
    s_inv = build_sqrtA_inv(grid)
    return apply_similarity(b, s_inv)


def build_loss_D_physical(grid: YeeGrid, gamma: float) -> sparse.csr_matrix:
    """Término D en variables físicas x para ω A x = (B + D) x.

    Con la convención temporal usada en el código para pérdidas decaying
    Im(ω)<0, se usa D = -i γ /(ε∞ ωp²) en el bloque V.
    """
    n2 = grid.n_dof
    ntot = grid.total_dof

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    if gamma <= 0:
        return sparse.csr_matrix((ntot, ntot), dtype=np.complex128)

    if grid.mode == Polarization.TM:
        v0 = block_offset(TMBlocks.BLK_VZ, n2)
        for k in range(n2):
            wp = grid.omega_p_mask[k]
            eps = grid.eps_inf_mask[k]
            if grid.metal_mask[k] and wp > 0:
                idx = v0 + k
                rows.append(idx)
                cols.append(idx)
                data.append(-1j * gamma / (eps * wp**2))

    else:
        vx0 = block_offset(TEBlocks.BLK_VX, n2)
        vy0 = block_offset(TEBlocks.BLK_VY, n2)

        for k in range(n2):
            wp_x = grid.omega_p_x[k]
            if grid.metal_x[k] and wp_x > 0:
                idx = vx0 + k
                rows.append(idx)
                cols.append(idx)
                data.append(-1j * gamma / (grid.eps_x[k] * wp_x**2))

            wp_y = grid.omega_p_y[k]
            if grid.metal_y[k] and wp_y > 0:
                idx = vy0 + k
                rows.append(idx)
                cols.append(idx)
                data.append(-1j * gamma / (grid.eps_y[k] * wp_y**2))

    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(ntot, ntot), dtype=np.complex128
    )


def build_loss_perturbation(grid: YeeGrid, gamma: float) -> sparse.csr_matrix:
    """Término de pérdida en la base Hermitiana y.

    Devuelve δH = A^{-1/2} D A^{-1/2}.
    """
    d_phys = build_loss_D_physical(grid, gamma)
    s_inv = build_sqrtA_inv(grid)
    return apply_similarity(d_phys, s_inv)


def assemble_systems(
    grid: YeeGrid, kx: float, ky: float, gamma: float = 0.0
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Retorna (Ĥ0, Ĥ_lossy, A_diagonal)."""
    h0 = build_H_hat(grid, kx, ky)
    a_diag = build_A_diagonal(grid)

    if gamma > 0:
        delta_h = build_loss_perturbation(grid, gamma)
        h_lossy = (h0 + delta_h).tocsr()
    else:
        h_lossy = h0

    return h0, h_lossy, a_diag
