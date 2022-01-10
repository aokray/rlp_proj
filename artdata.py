import numpy as np
from nptyping import NDArray
from typing import List

class MVAGenerator:
    '''
    MVAGenerator takes a dataset (without label), and optionally a pair of view index sets.
    If the pair of view index sets is not provided, an even (or approximately even)
    feature split will be used.

    ----------------- WARNING -----------------\n
    This class assumes that |viewset 1| = |viewset 2|
    '''
    def __init__(self, X : NDArray, class_outlier_rate : float, attr_outlier_rate : float, class_attr_outlier_rate : float, v1_v2 : List[NDArray] = None):
        self.X = X
        self.n, self.p = X.shape
        self.class_outlier_rate = class_outlier_rate
        self.attr_outlier_rate = attr_outlier_rate
        self.class_attr_outlier_rate = class_attr_outlier_rate
        self.v1_v2 = v1_v2

        self.class_outlier_idxs = None
        self.attr_outlier_idxs = None
        self.class_attr_outlier_idxs = None
        self.Y = np.zeros((self.n, 1))

    def _generate_class_outlier(self) -> None:
        # Generates class outliers, assigns label 1
        self.class_outlier_idxs = np.random.choice(self.n, int(self.class_outlier_rate * self.n), replace=False)
        other_idxs = np.setdiff1d(np.arange(self.n), self.class_outlier_idxs)
        
        # This is one reason |viewset 1| must equal |viewset 2|
        self.X = np.block([
            [self.X[self.class_outlier_idxs][:,self.v1_v2[1]], self.X[self.class_outlier_idxs][:,self.v1_v2[0]]],
            [self.X[other_idxs][:,self.v1_v2[0]], self.X[other_idxs][:,self.v1_v2[1]]]
        ])

        self.Y[self.class_outlier_idxs] = 1

    # TODO: Implement this
    def _generate_attr_outlier(self) -> None:
        # Generates attr outliers, assigns label 2
        self.attr_outlier_idxs = np.random.choice(self.n, int(self.attr_outlier_rate * self.n), replace=False)

    # TODO: Implement this
    def _generate_class_attr_outlier(self) -> None:
        # Generates class attr outliers, assigns label 3
        self.class_attr_outlier_idxs = np.random.choice(self.n, int(self.class_attr_outlier_rate * self.n), replace=False)

    def generate(self) -> NDArray:
        if self.v1_v2 is None:
            self.v1_v2 = [np.arange(int(self.p/2)), np.arange(int(self.p/2), self.p)]

        if self.class_outlier_rate:
            self._generate_class_outlier()

        if self.attr_outlier_rate:
            self._generate_attr_outlier()

        if self.class_attr_outlier_rate:
            self._generate_class_attr_outlier()

        return self.X, self.Y