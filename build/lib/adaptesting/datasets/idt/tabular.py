import numpy as np
import torch
from typing import Tuple, Optional

from ..base import TabularIDTDataset


def G(x):
    if -1 < x < -0.5:
        return np.exp(-1 / (1 - (4 * x + 3) ** 2))
    if -0.5 < x < 0:
        return -np.exp(-1 / (1 - (4 * x + 1) ** 2))
    return 0.0


def rejection_sampler(seed, density, d, density_max, number_samples, x_min, x_max):
    samples = []
    count = 0
    rs = np.random.RandomState(seed)
    while count < number_samples:
        x = rs.uniform(x_min, x_max, d)
        y = rs.uniform(0, density_max)
        if y <= density(x):
            count += 1
            samples.append(x)
    return np.array(samples)


def f_theta(x, p, s, perturbation_multiplier=1, seed=None):
    x = np.atleast_1d(x)
    d = x.shape[0]
    assert perturbation_multiplier * p ** (-s) * np.exp(-d) <= 1, "density is negative"
    np.random.seed(seed)
    theta = np.random.choice([-1, 1], p ** d)
    output = 0.0
    grid = np.array(np.meshgrid(*[np.arange(1, p + 1) for _ in range(d)])).T.reshape(-1, d)
    for i, cell in enumerate(grid):
        output += theta[i] * np.prod([G(x[r] * p - cell[r]) for r in range(d)])
    output *= p ** (-s) * perturbation_multiplier
    if np.min(x) >= 0 and np.max(x) <= 1:
        output += 1
    np.random.seed(None)
    return output


def f_theta_sampler(f_theta_seed, sampling_seed, number_samples, p, s, perturbation_multiplier, d):
    density_max = 1 + perturbation_multiplier * p ** (-s) * np.exp(-d)
    return rejection_sampler(
        sampling_seed,
        lambda x: f_theta(x, p, s, perturbation_multiplier, f_theta_seed),
        d,
        density_max,
        number_samples,
        0,
        1,
    )


class SyntheticJointSplit(TabularIDTDataset):
    """
    Synthetic dependence dataset based on the AggInc f_theta sampler construction.
    """

    def __init__(
        self,
        root: str = './data',
        N: int = 500,
        download: bool = True,
        p: int = 2,
        s: float = 1.0,
        d: int = 2,
        dx: int = 1,
        perturbation_multiplier: Optional[float] = None,
        mult: float = 2.0,
        f_theta_seed: int = 0,
        sampling_seed: Optional[int] = None,
        **kwargs
    ):
        self.p = p
        self.s = s
        self.d = d
        self.dx = dx
        self.mult = mult
        self.f_theta_seed = f_theta_seed
        self.sampling_seed = sampling_seed
        self.perturbation_multiplier = perturbation_multiplier
        super().__init__(root, N, download, **kwargs)

    def _download(self):
        pass

    def _check_exists(self) -> bool:
        return True

    def _sample_joint(self, sampling_seed):
        perturbation_multiplier = self.perturbation_multiplier
        if perturbation_multiplier is None:
            perturbation_multiplier = np.exp(self.d) * self.p ** self.s / self.mult

        return f_theta_sampler(
            self.f_theta_seed,
            sampling_seed,
            self.N,
            self.p,
            self.s,
            perturbation_multiplier,
            self.d,
        )

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 0 < self.dx < self.d:
            raise ValueError("dx must satisfy 0 < dx < d")

        sampling_seed = self.seed if self.sampling_seed is None else self.sampling_seed
        Z = self._sample_joint(sampling_seed)
        X = Z[:, :self.dx]

        if self.t1_check:
            Z_independent = self._sample_joint(sampling_seed + 10_000)
            Y = Z_independent[:, self.dx:]
        else:
            Y = Z[:, self.dx:]

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None:
            Y = self.target_transform(Y)

        return X, Y

__all__ = [
    'SyntheticJointSplit',
]
