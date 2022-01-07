import numpy as np


def _gaussian_mat(p, k, params) -> np.ndarray:
    return np.random.normal(**params, size=(p, k))


def _sparse_mat(p: int, k: int, const=np.sqrt(3)) -> np.ndarray:
    r = np.zeros((p, k))

    for i in range(p):
        for j in range(k):
            chance = np.random.uniform()

            if chance < (1 / 6):
                r[i, j] = const
            elif chance < (1 / 3):
                r[i, j] = -const

    return r


class RLP:
    def __init__(self, k: int, random_mode: str = "gaussian"):
        self.k = k
        self.random_mode = random_mode
        self.proj_mat = None

    def fit(self, X: np.ndarray, **kw_args):
        """
        Makes a random projection matrix on EACH CALL, kw_args can be used with the following random_mode's:
            gaussian: mean, sd
            sparse: const
        """
        n, p = X.shape
        temp_dict = {}

        if self.random_mode == "gaussian":
            if "mean" in kw_args.keys():
                temp_dict["loc"] = kw_args["mean"]

            if "sd" in kw_args.keys():
                temp_dict["scale"] = kw_args["sd"]

            self.proj_mat = _gaussian_mat(p, self.k, temp_dict)
        elif self.random_mode == "sparse":
            if "const" in kw_args.keys():
                self.proj_mat = _sparse_mat(p, self.k, kw_args["const"])
            else:
                self.proj_mat = _sparse_mat(p, self.k)
        else:
            raise Exception("Unknown random_mode for generating random matrices")

    def transform(self, X: np.ndarray):
        """
        Expects X to be n x p
        """

        return X @ self.proj_mat

    def fit_transform(self, x, **kw_args):
        self.fit(x, **kw_args)
        return self.transform(x)


# Exmple usage:
# x = ... # (in n x p, for p > 2)
# r = RLP(2)
# transformed_data = r.fit_transform(x, mean=1, sd=4)
# transformed_data in (n x k, here k= 2)
