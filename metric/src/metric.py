import numpy as np
from scipy.stats import multivariate_normal


def JSD(raw_latent, swap_latent):
    raw_mean = np.mean(raw_latent, axis=0)
    raw_cov = np.cov(raw_latent, rowvar=False)

    swap_mean = np.mean(raw_latent, axis=0)
    swap_cov = np.cov(swap_latent, rowvar=False)

    p = multivariate_normal(mean=raw_mean, cov=raw_cov)
    q = multivariate_normal(mean=swap_mean, cov=swap_cov)

    n = 10000

    X = p.rvs(n)
    p_X = p.pdf(X)
    q_X = q.pdf(X)
    m_X = 0.5 * (p_X + q_X)
    r_pm = np.log2(p_X / m_X)

    X = q.rvs(n)
    p_X = p.pdf(X)
    q_X = q.pdf(X)
    m_X = 0.5 * (p_X + q_X)
    r_qm = np.log2(q_X / m_X)

    kl_pm = np.mean(r_pm)
    kl_qm = np.mean(r_qm)

    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return jsd


def MSE(raw_latent, swap_latent):
    mse = np.mean(np.square(raw_latent - swap_latent))
    return mse

if __name__ == "__main__":
    raw_latent = np.random.rand(100, 32)
    swap_latent = np.random.rand(100, 32)

    jsd = JSD(raw_latent, swap_latent)
    mse = MSE(raw_latent, swap_latent)

    print("JSD: ", jsd)
    print("MSE: ", mse)