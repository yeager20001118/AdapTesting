"""
Tabular datasets for two-sample testing using real datasets.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Tuple

from .base import TabularTSTDataset


class HiggsBoson(TabularTSTDataset):
    """
    Higgs Boson dataset: signal vs background events.
    """

    filename = "higgs_sample.csv"

    def __init__(
        self,
        root: str = './data',
        N: int = 50000,
        M: int = 50000,
        download: bool = True,
        use_low_level_features: bool = True,
        **kwargs
    ):
        self.use_low_level_features = use_low_level_features
        super().__init__(root, N, M, download, **kwargs)

    def _download(self):
        filepath = os.path.join(self.root, self.filename)
        if not os.path.exists(filepath):
            print("Downloading first 100k samples from official Higgs dataset...")
            try:
                self._download_higgs_sample(filepath)
                print("Downloaded successfully!")
            except Exception as e:
                raise RuntimeError(f"Could not download Higgs dataset: {e}")

    def _download_higgs_sample(self, output_path):
        import urllib.request
        import gzip

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
        max_samples = 100000

        print(f"Streaming first {max_samples} samples from Higgs dataset...")

        with urllib.request.urlopen(url) as response:
            with gzip.open(response, 'rt') as gz_file:
                with open(output_path, 'w') as out_file:
                    for i, line in enumerate(gz_file):
                        if i >= max_samples:
                            break
                        out_file.write(line)
                        if (i + 1) % 10000 == 0:
                            print(f"Downloaded {i + 1} samples...")

        print(f"Successfully downloaded {min(max_samples, i + 1)} samples!")

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root, self.filename))

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        filepath = os.path.join(self.root, self.filename)
        data = pd.read_csv(filepath, header=None)

        labels = data.iloc[:, 0].values
        features = data.iloc[:, 1:].values

        if not self.use_low_level_features:
            features = features[:, :21]

        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        signal_indices = np.where(labels == 1)[0]
        background_indices = np.where(labels == 0)[0]

        np.random.seed(self.seed)

        signal_sample_indices = np.random.choice(signal_indices, min(self.N, len(signal_indices)), replace=False)
        background_sample_indices = np.random.choice(background_indices, min(self.M, len(background_indices)), replace=False)

        X_samples = features[signal_sample_indices]
        Y_samples = features[background_sample_indices]

        if self.t1_check:
            print("Type-I error check: Drawing both X and Y from background distribution")
            total_needed = self.N + self.M
            if len(background_indices) >= total_needed:
                combined_indices = np.random.choice(background_indices, total_needed, replace=False)
            else:
                combined_indices = np.random.choice(background_indices, total_needed, replace=True)

            X_samples = features[combined_indices[:self.N]]
            Y_samples = features[combined_indices[self.N:self.N + self.M]]

        return torch.tensor(X_samples, dtype=torch.float32), torch.tensor(Y_samples, dtype=torch.float32)


class HDGM(TabularTSTDataset):
    """
    Hierarchical Dirichlet Gaussian Mixture (HDGM) synthetic dataset.
    """

    def __init__(
        self,
        root: str = './data',
        N: int = 500,
        M: int = 500,
        download: bool = True,
        d: int = 10,
        n_clusters: int = 2,
        level: str = "hard",
        kk: int = 0,
        **kwargs
    ):
        self.d = d
        self.n_clusters = n_clusters
        self.level = level
        self.kk = kk
        super().__init__(root, N, M, download, **kwargs)

    def _download(self):
        pass

    def _check_exists(self) -> bool:
        return True

    def _generate_hdgm_cov_matrix(self, n_clusters, d, cluster_gap):
        mu_mx = np.zeros([n_clusters, d])
        for i in range(n_clusters):
            mu_mx[i] = mu_mx[i] + cluster_gap * i
        sigma_mx_1 = np.eye(d)
        sigma_mx_2 = [np.eye(d), np.eye(d)]
        sigma_mx_2[0][0, 1] = 0.5
        sigma_mx_2[0][1, 0] = 0.5
        sigma_mx_2[1][0, 1] = -0.5
        sigma_mx_2[1][1, 0] = -0.5
        return mu_mx, sigma_mx_1, sigma_mx_2

    def _sample_hdgm(self, N, M, d, n_clusters, kk, level, t1_check):
        if level == "hard":
            mu_mx_1, sigma_mx_1, sigma_mx_2 = self._generate_hdgm_cov_matrix(n_clusters, d, 0.5)
            mu_mx_2 = mu_mx_1
        elif level == "medium":
            mu_mx_1, sigma_mx_1, sigma_mx_2 = self._generate_hdgm_cov_matrix(n_clusters, d, 10)
            mu_mx_2 = mu_mx_1
        else:
            mu_mx_1, sigma_mx_1, sigma_mx_2 = self._generate_hdgm_cov_matrix(n_clusters, d, 10)
            mu_mx_2 = mu_mx_1 - 1.5

        P = np.zeros([N * n_clusters, d])
        Q = np.zeros([M * n_clusters, d])

        if t1_check:
            for i in range(n_clusters):
                np.random.seed(seed=1102 * kk + i + N + M)
                P[i * N:(i + 1) * N, :] = np.random.multivariate_normal(mu_mx_1[i], sigma_mx_1, N)
                np.random.seed(seed=819 * kk + i + N + M)
                Q[i * M:(i + 1) * M, :] = np.random.multivariate_normal(mu_mx_1[i], sigma_mx_1, M)
        else:
            for i in range(n_clusters):
                np.random.seed(seed=1102 * kk + i + N + M)
                P[i * N:(i + 1) * N, :] = np.random.multivariate_normal(mu_mx_1[i], sigma_mx_1, N)
                np.random.seed(seed=819 * kk + i + N + M)
                Q[i * M:(i + 1) * M, :] = np.random.multivariate_normal(mu_mx_2[i], sigma_mx_2[i], M)

        idx_P = np.random.choice(len(P), N, replace=False)
        idx_Q = np.random.choice(len(Q), M, replace=False)
        return P[idx_P], Q[idx_Q]

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        P, Q = self._sample_hdgm(self.N, self.M, self.d, self.n_clusters, self.kk, self.level, self.t1_check)
        return torch.tensor(P, dtype=torch.float32), torch.tensor(Q, dtype=torch.float32)


class BLOB(TabularTSTDataset):
    """
    BLOB synthetic dataset.
    """

    def __init__(
        self,
        root: str = './data',
        N: int = 500,
        M: int = 500,
        download: bool = True,
        rows: int = 3,
        cols: int = 3,
        var: float = 0.03,
        min_corr: float = 0.02,
        kk: int = 0,
        **kwargs
    ):
        self.rows = rows
        self.cols = cols
        self.var = var
        self.min_corr = min_corr
        self.kk = kk
        super().__init__(root, N, M, download, **kwargs)

    def _download(self):
        pass

    def _check_exists(self) -> bool:
        return True

    def _create_grid(self, n_rows, n_cols):
        return np.array([[i, j] for i in range(n_rows) for j in range(n_cols)])

    def _create_blob_cov_matrix(self, n_locs, variance, min_corr):
        n_side = n_locs // 2
        correlations = min_corr + np.arange(n_side) * 0.002
        correlations = np.concatenate([
            correlations[::-1] * -1,
            [0],
            correlations
        ])
        return np.array([
            [[variance, corr], [corr, variance]]
            for corr in correlations
        ]).round(4)

    def _sample_blob(self, N_more, N_less, rs, is_alternative):
        mu = np.zeros(2)
        sigma = np.eye(2) * (self.var - 0.01)
        sigmas = self._create_blob_cov_matrix(self.rows * self.cols, self.var, self.min_corr)
        random_state = np.random.RandomState(rs)

        X = random_state.multivariate_normal(mu, sigma, size=N_more)
        X_row = random_state.randint(self.rows, size=N_more)
        X_col = random_state.randint(self.cols, size=N_more)
        X[:, 0] += X_row
        X[:, 1] += X_col

        if is_alternative:
            Y = random_state.multivariate_normal(mu, np.eye(2), size=N_less)
            Y_row = random_state.randint(self.rows, size=N_less)
            Y_col = random_state.randint(self.cols, size=N_less)
            locs = self._create_grid(self.rows, self.cols)
            for i, loc in enumerate(locs):
                tgt_row, tgt_col = loc
                L = np.linalg.cholesky(sigmas[i])
                mask = (Y_row == tgt_row) & (Y_col == tgt_col)
                Y[mask] = Y[mask] @ L + loc
        else:
            Y = random_state.multivariate_normal(mu, sigma, size=N_less)
            Y[:, 0] += random_state.randint(self.rows, size=N_less)
            Y[:, 1] += random_state.randint(self.cols, size=N_less)

        return X, Y

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X, Y = self._sample_blob(self.N, self.M, self.seed + self.kk, not self.t1_check)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


__all__ = [
    'HiggsBoson',
    'HDGM',
    'BLOB',
]
