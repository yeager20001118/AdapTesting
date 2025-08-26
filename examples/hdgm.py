import numpy as np
def generate_hdgm_cov_matrix(n_clusters, d, cluster_gap):
    mu_mx = np.zeros([n_clusters, d])
    for i in range(n_clusters):
        mu_mx[i] = mu_mx[i] + cluster_gap*i
    sigma_mx_1 = np.eye(d)
    sigma_mx_2 = [np.eye(d), np.eye(d)]
    sigma_mx_2[0][0, 1] = 0.5
    sigma_mx_2[0][1, 0] = 0.5
    sigma_mx_2[1][0, 1] = -0.5
    sigma_mx_2[1][1, 0] = -0.5

    return mu_mx, sigma_mx_1, sigma_mx_2

def sample_hdgm(N, M, d=10, n_clusters=2, kk=0, level="hard", t1_check=False):
    if level == "hard":
        mu_mx_1, sigma_mx_1, sigma_mx_2 = generate_hdgm_cov_matrix(n_clusters, d, 0.5)
        mu_mx_2 = mu_mx_1
    elif level == "medium":
        mu_mx_1, sigma_mx_1, sigma_mx_2 = generate_hdgm_cov_matrix(n_clusters, d, 10)
        mu_mx_2 = mu_mx_1
    else:
        mu_mx_1, sigma_mx_1, sigma_mx_2 = generate_hdgm_cov_matrix(n_clusters, d, 10)
        mu_mx_2 = mu_mx_1 - 1.5

    P = np.zeros([N*n_clusters, d])
    Q = np.zeros([M*n_clusters, d])

    if t1_check:
        for i in range(n_clusters):
            np.random.seed(seed=1102*kk + i + N + M)
            P[i*N:(i+1)*N, :] = np.random.multivariate_normal(mu_mx_1[i], sigma_mx_1, N)
            np.random.seed(seed=819*kk + i + N + M)
            Q[i*M:(i+1)*M, :] = np.random.multivariate_normal(mu_mx_1[i], sigma_mx_1, M)
    else:
        for i in range(n_clusters):
            np.random.seed(seed=1102*kk + i + N + M)
            P[i*N:(i+1)*N, :] = np.random.multivariate_normal(mu_mx_1[i], sigma_mx_1, N)
            np.random.seed(seed=819*kk + i + N + M)
            Q[i*M:(i+1)*M, :] = np.random.multivariate_normal(mu_mx_2[i], sigma_mx_2[i], M)

    idx_P = np.random.choice(len(P), N, replace=False)
    idx_Q = np.random.choice(len(Q), M, replace=False)
    return P[idx_P], Q[idx_Q]

P, Q = sample_hdgm(500, 500, level="easy")

import matplotlib.pyplot as plt
plt.plot(P[:, 0], P[:, 1], 'o', label="Distribution P")
plt.plot(Q[:, 0], Q[:, 1], 'o', label="Distribution Q")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("HDGM Dataset (level: easy)")
plt.legend()
plt.savefig('hdgm_plot.png')
