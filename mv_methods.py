import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels, paired_cosine_distances
from sklearn.metrics import roc_auc_score
from scipy.linalg import eigh, inv
from sklearn.cluster import AffinityPropagation


def _L_matrix(X, n_X, n_feats):
    diff_Matrix = np.zeros((n_X, n_X, n_feats))
    for i in range(n_X):
        for j in range(n_X):
            diff_Matrix[i, j] = X[i] - X[j]
    L = -np.square(np.linalg.norm(diff_Matrix, 2, 2))
    med = np.median(L)
    np.fill_diagonal(L, med)

    return L

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


def AffProp(X_v1, X_v2, Y):
    n = X_v1.shape[0]
    p1 = X_v1.shape[1]
    p2 = X_v2.shape[1]

    L1 = _L_matrix(X_v1, n, p1)
    L2 = _L_matrix(X_v2, n, p2)
    ap = AffinityPropagation(random_state=None)
    C1 = ap.fit_predict(X_v1)
    C2 = ap.fit_predict(X_v2)
    Z1 = np.zeros([n, n])
    Z2 = np.zeros([n, n])

    for i in range(n):  # number of z's
        for j in range(n):  # elements of each vector z
            if i == j:
                Z1[i, j] = 0
                Z2[i, j] = 0
            else:
                Z1[i, j] = np.exp(L1[i, C1[j]] + L1[i, j] - 2)
                Z2[i, j] = np.exp(L2[i, C2[j]] + L2[i, j] - 2)
        Z1[i] = Z1[i] / np.sum(Z1[i])
        Z2[i] = Z2[i] / np.sum(Z2[i])

    # anomaly score
    AD = 1 / np.square(np.linalg.norm(Z1 - Z2, 2, 1))

    auc = roc_auc_score(Y, AD)

    return auc