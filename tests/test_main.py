from adaptesting import tst

import torch
# from main import *
import random
from torch.distributions import MultivariateNormal
import time

start = time.time()
mean = torch.tensor([0.5, 0.5])
cov1 = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
cov2 = torch.tensor([[1.0, 0], [0, 1.0]])
mvn1 = MultivariateNormal(mean, cov1)
mvn2 = MultivariateNormal(mean, cov2)

torch.manual_seed(0)
random.seed(0)

counter = 0
n_trial = 100
for _ in range(n_trial):

    Z1 = mvn1.sample((1000,))
    Z2 = mvn2.sample((1000,))  # Test power
    # Z2 = mvn1.sample((1000,))  # Type-I error

    # Create a list of indices from 0 to 199
    indices = list(range(1000))

    # Shuffle the indices
    random.shuffle(indices)

    # Select the first 100 shuffled indices for X
    X_indices = indices[:250]

    # Select the remaining indices for Y
    # Y_indices = indices[50:]

    # Sample X and Y from Z using the selected indices
    X = Z1[X_indices]
    Y = Z2[X_indices]

    # print(X[:5], Y[:5])
    # h, _ = tst(X, Y, device = "cuda") # default method is median heuristic
    h, _ = tst(X, Y, device="cuda", kernel="laplace_gaussian",
               method="agg", seed=42)
    counter += h
    break

print(f"Power: {counter}/{n_trial}")
end = time.time()
print(f"Time taken: {end - start:.4f} seconds")
