import numpy as np
import matplotlib.pyplot as plt
from rlp import RLP
from mv_methods import HOAD
from sklearn.metrics import roc_auc_score


def random_split(p, requires_equal=False):
    cand_idxs = np.arange(p)

    # Choose a random feature split such that there is at least one feature
    # in both views (thus the length range [1,p-1])
    if requires_equal:
        if p % 2:
            raise Exception(
                f"Cannot create 2 equal sized views from original sized feature set of {p}"
            )
        num_v1 = int(p / 2)
    else:
        num_v1 = np.random.randint(1, p - 1)

    v1_idxs = np.random.choice(cand_idxs, num_v1, replace=False)

    v2_idxs = np.setdiff1d(cand_idxs, v1_idxs)

    return v1_idxs, v2_idxs


data = np.loadtxt("diabetes.csv", delimiter=",", skiprows=1)

x = data[:, :-1]
y = data[:, -1]

rounds = 10
k = 10
lmbda = 100

hoad_rlp_aucs = []
hoad_rand_aucs = []

for i in range(rounds):
    print(f"Round {i}...........")
    # Make two viewsets from the original dataset, using two seperate random linear projections
    v1_proj = RLP(4, "gaussian")
    v2_proj = RLP(4, "sparse")
    rlp_X_v1 = v1_proj.fit_transform(x)
    rlp_X_v2 = v2_proj.fit_transform(x)

    # Get random view splits
    idxs1, idxs2 = random_split(x.shape[1], True)
    rand_X_v1 = x[:, idxs1]
    rand_X_v2 = x[:, idxs2]

    # Do anomaly detection
    hoad_rlp_scores = HOAD(rlp_X_v1, rlp_X_v2, k, [k], lmbda)
    hoad_rlp_aucs.append(roc_auc_score(y, hoad_rlp_scores))
    print(f"\tRLP AUC Score: {roc_auc_score(y, hoad_rlp_scores)}")

    hoad_rand_scores = HOAD(rand_X_v1, rand_X_v2, k, [k], lmbda)
    hoad_rand_aucs.append(roc_auc_score(y, hoad_rand_scores))
    print(f"\tRandom View Split AUC Score: {roc_auc_score(y, hoad_rand_scores)}")


print(
    f"Average AUC Results:\nMethod:\tRLP: \t Random:\nHOAD\t{np.average(hoad_rlp_aucs):>.4f}\t{np.average(hoad_rand_aucs):>.4f}"
)
plt.bar(
    range(2),
    [np.average(hoad_rlp_aucs), np.average(hoad_rand_aucs)],
    yerr=[np.std(hoad_rlp_aucs), np.std(hoad_rand_aucs)],
    capsize=4
)
plt.xticks(range(2), ["RLP", "Random View Split"], rotation=45)
plt.show()
