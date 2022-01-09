import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels, paired_cosine_distances
from scipy.linalg import eigh, inv


def HOAD(v1, v2, k, keval, m):
    SimV1 = pairwise_kernels(v1, v2, "cosine")
    SimV2 = pairwise_kernels(v2, v1, "cosine")

    n = SimV1.shape[0]

    Z = np.block([[SimV1, m * np.identity(n)], [m * np.identity(n), SimV2]])

    D = np.diag(np.sum(Z, axis=1))

    L = D - Z

    w, vr = eigh(L, eigvals=(0, k - 1))

    adscore = np.zeros((n, len(keval)))

    for i in range(len(keval)):

        Hv1 = vr[0:n, 0 : keval[i]]
        Hv2 = vr[n:, 0 : keval[i]]

        adscore[:, i] = 1 - paired_cosine_distances(Hv1, Hv2)

    return adscore
