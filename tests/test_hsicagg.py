from agginc import agginc
import numpy as np
import time
import itertools

def G(x):
    """
    Function G defined in Section 5.4 of our paper.
    input: x: real number
    output: G(x): real number
    """
    if -1 < x and x < -0.5:
        return np.exp(-1 / (1 - (4 * x + 3) ** 2))
    if -0.5 < x and x < 0:
        return -np.exp(-1 / (1 - (4 * x + 1) ** 2))
    return 0

def rejection_sampler(seed, density, d, density_max, number_samples, x_min, x_max):
    """
    Sample from density using a rejection sampler.
    inputs: seed: integer random seed
            density: probability density function
            d: dimension of input of the density
            density_max: maximum of the density
            number_samples: number of samples
            x_min: density is 0 on (-\infty,x_min)^d
            x_max: density is 0 on (x_max,\infty)^d
    output: number_samples samples from density sampled from [x_min, x_max]^d
    """
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
    """
    Function f_theta defined in in Section 5.4 (Eq. (17)) of our paper.
    inputs: x: (d,) array (point in R^d)
            p: non-negative integer (number of permutations)
            s: positive number (smoothness parameter of Sobolev ball (Eq. (1))
            perturbation_multiplier: positive number (c_d in Eq. (17))
            seed: integer random seed (samples theta in Eq. (17))
    output: real number f_theta(x)
    """
    x = np.atleast_1d(x)
    d = x.shape[0]
    assert perturbation_multiplier * p ** (-s) * np.exp(-d) <= 1, "density is negative"
    np.random.seed(seed)
    theta = np.random.choice([-1, 1], p**d)
    output = 0
    I = list(itertools.product([i + 1 for i in range(p)], repeat=d))  # set {1,...,p}^d
    for i in range(len(I)):
        j = I[i]
        output += theta[i] * np.prod([G(x[r] * p - j[r]) for r in range(d)])
    output *= p ** (-s) * perturbation_multiplier
    if np.min(x) >= 0 and np.max(x) <= 1:
        output += 1
    np.random.seed(None)
    return output

def f_theta_sampler(
    f_theta_seed, sampling_seed, number_samples, p, s, perturbation_multiplier, d
):
    """
    Sample from the probability density function f_theta.
    inputs: f_theta_seed: integer random seed for f_theta
            sampling_seed: integer random seed for rejection sampler
            number_samples: number of samples
            p: non-negative integer (number of permutations)
            s: positive number (smoothness parameter of Sobolev ball (Eq. (1))
            perturbation_multiplier: positive number (c_d in Eq. (17))
            non-negative integer (dimension of input of density)
    output: number_samples samples from f_theta
    """
    density_max = 1 + perturbation_multiplier * p ** (-s) * np.exp(
        -d
    )  # maximum of f_theta
    return rejection_sampler(
        sampling_seed,
        lambda x: f_theta(x, p, s, perturbation_multiplier, f_theta_seed),
        d,
        density_max,
        number_samples,
        0,
        1,
    )

f_theta_seed = 0
p = 2 
s = 1 # useless with our scaling
d = 2 # dx = 1 and dy = 1
rep = 100
mult = 2 

tests_hsic_vary_n = [agginc] # [nfsic, hsicagginc1, hsicagginc100, hsicagginc200, hsicaggincquad]
tests = tests_hsic_vary_n

from adaptesting import idt
import torch

method_names = ["hsicagg", "rdc", "fsic"]

# N_values = [200, 400, 600, 800, 1000]
N_values=[1000]
outputs_hsic_vary_n = np.zeros((len(method_names), len(N_values), rep))
stats_hsic_vary_n = np.zeros((len(method_names), len(N_values), rep))
p_values_hsic_vary_n = np.zeros((len(method_names), len(N_values), rep))
rs = np.random.RandomState(0)
seed = 0

for i in range(rep):
    t0 = time.time()
    for r in range(len(N_values)):
        N = N_values[r]
        seed += 1
        perturbation_multiplier = np.exp(d) * p ** s / mult
        # Test power
        Z = f_theta_sampler(f_theta_seed, seed, N, p, s, perturbation_multiplier, d)
        X = np.expand_dims(Z[:, 0], 1)
        Y = np.expand_dims(Z[:, 1], 1)

        # Type-I error 
        # Zx = f_theta_sampler(f_theta_seed, seed, N, p, s, perturbation_multiplier, d)
        # Zy = f_theta_sampler(f_theta_seed, seed + 10_000, N, p, s, perturbation_multiplier, d)
        # X = np.expand_dims(Zx[:, 0], 1)
        # Y = np.expand_dims(Zy[:, 1], 1)

        X = torch.tensor(X, device="cuda")
        Y = torch.tensor(Y, device="cuda")
        for j in range(len(method_names)):
            method = method_names[j]
            if method == "hsicagg":
                h, stat_value, p_value = idt(
                    X, Y, device="cuda", method=method, alpha=0.05,
                    R=X.shape[0]-1, seed=seed, n_perm=500
                )
            else:
                h, stat_value, p_value = idt(
                    X, Y, device="cuda", method=method, alpha=0.05,
                    seed=seed, n_perm=500
                )
            outputs_hsic_vary_n[j][r][i] = h
            stats_hsic_vary_n[j][r][i] = stat_value
            p_values_hsic_vary_n[j][r][i] = p_value
    print(i + 1, "/", rep, "time:", time.time() - t0)
power = np.array([[np.mean(outputs_hsic_vary_n[j][r])for r in range(len(N_values))] for j in range(len(method_names))])
print(power)
for j, method in enumerate(method_names):
    print(method, "mean_stat:", [np.mean(stats_hsic_vary_n[j][r]) for r in range(len(N_values))])
    print(method, "mean_p_value:", [np.mean(p_values_hsic_vary_n[j][r]) for r in range(len(N_values))])
# np.save("hsic_vary_n.npy", power)
# np.save("hsic_vary_n_x_axis.npy", N_values)
