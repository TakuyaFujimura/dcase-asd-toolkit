from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA

from asdkit.utils.dcase_utils import MACHINE_DICT


class PCA_GEVD_NRFT:
    def __init__(
        self,
        pca_dim: int = 64,
        n_components: int = 16,
        eps_scale: float = 0,
        save_eigen_stats: bool = False,
    ):
        self.pca_dim = pca_dim
        self.n_components = n_components
        self.eps_scale = eps_scale
        self.save_eigen_stats = save_eigen_stats

    def __call__(
        self,
        embed: np.ndarray,
        embed_sup: np.ndarray,
        machine: str,
    ) -> np.ndarray:
        if machine in MACHINE_DICT["dcase2025-noise"]:
            W, _, pca = estimate_subspace_pca_gevd(
                A=embed,
                B=embed_sup,
                pca_dim=self.pca_dim,
                n_components=self.n_components,
                eps_scale=self.eps_scale,
            )
        else:
            assert machine in MACHINE_DICT["dcase2025-clean"]
            W, pca = estimate_subspace_pca(
                embed=embed_sup,
                n_components=self.n_components,
            )

        return (embed - pca.mean_) @ W


def estimate_subspace_pca(
    embed: np.ndarray, n_components: int = 16
) -> Tuple[np.ndarray, PCA]:
    if embed.ndim != 2:
        raise ValueError(f"embed must be a 2-D array, got {embed.shape}")

    n_components = min(n_components, embed.shape[0], embed.shape[1])
    pca = PCA(n_components=n_components, whiten=False)
    pca.fit(embed)

    W = pca.components_.T
    return W, pca


def estimate_subspace_pca_gevd(
    A: np.ndarray,
    B: np.ndarray,
    pca_dim: int = 64,
    n_components: int = 16,
    eps_scale: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Args:
        A: [B1, D]
        B: [B2, D]
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2-D arrays, " f"got {A.shape} and {B.shape}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            "A and B feature dims differ: " f"{A.shape[1]} != {B.shape[1]}"
        )

    Z = np.concatenate([A, B], axis=0)
    pca_dim = min(pca_dim, Z.shape[0], Z.shape[1])
    n_components = min(n_components, pca_dim)

    # PCA #################################################
    # mu = mean(Z)
    # A = A - mu, B = B - mu
    # obtain the PCA projection matrix V
    # A = A @ V, B = B @ V
    pca = PCA(n_components=pca_dim, whiten=False)
    Z_pca = pca.fit_transform(Z)
    A_pca = Z_pca[: len(A)]
    B_pca = Z_pca[len(A) :]

    # GEVD #################################################
    # compute the covariance matrices A_cov and B_cov
    # solve A_cov * u = lambda * B_cov * u
    A_cov = np.atleast_2d(np.cov(A_pca, rowvar=False))
    B_cov = np.atleast_2d(np.cov(B_pca, rowvar=False))

    eps = eps_scale * np.trace(B_cov) / pca_dim
    assert eps >= 0
    B_cov = B_cov + eps * np.eye(pca_dim)

    eigvals, eigvecs = eigh(A_cov, B_cov)

    # Select eigenvectors
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[:n_components]
    U = eigvecs[:, :n_components]

    W = pca.components_.T @ U

    return W, eigvals, pca
