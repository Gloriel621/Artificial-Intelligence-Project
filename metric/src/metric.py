import torch
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

from model import *
from dataset import *


def JSD(raw_latent, swap_latent):
    raw_mean = np.mean(raw_latent, axis=0)
    raw_cov = np.cov(raw_latent, rowvar=False)

    swap_mean = np.mean(raw_latent, axis=0)
    swap_cov = np.cov(swap_latent, rowvar=False)

    p = multivariate_normal(mean=raw_mean, cov=raw_cov)
    q = multivariate_normal(mean=swap_mean, cov=swap_cov)

    n = 100000

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

def get_latent(input_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    deblurrer = NAFNet(3, 8, 1, [1, 1, 28], [1, 1, 1])
    deblurrer.load_state_dict(torch.load("metric/save/deblur.pth"))

    dataset = DeblurDataset(input_path, "test")
    latents = []

    deblurrer.to(device)
    deblurrer.eval()
    with torch.no_grad():
        for image in tqdm(dataset):
            latent = deblurrer.get_latent(image.unsqueeze(0).to(device))
            latents.append(latent.copy())

    latents = np.concatenate(latents, axis=0)
    return latents

if __name__ == "__main__":
    raw_latent = get_latent("metric/data/raw")
    swap_latent = get_latent("metric/data/swap_partial")

    jsd = JSD(raw_latent, swap_latent)
    mse = MSE(raw_latent, swap_latent)

    print("JSD: ", jsd)
    print("MSE: ", mse)

    swap_latent = get_latent("metric/data/swap_full")

    jsd = JSD(raw_latent, swap_latent)
    mse = MSE(raw_latent, swap_latent)

    print("JSD: ", jsd)
    print("MSE: ", mse)