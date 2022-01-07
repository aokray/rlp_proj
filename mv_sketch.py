import numpy as np
from rlp import RLP

# data = np.loadtxt('', delimiter=',')

# Make two viewsets from the original dataset, using two seperate random linear projections
v1_proj = RLP(3, 'gaussian')
v2_proj = RLP(3, 'sparse')

# X_v1 = v1_proj.fit_transform(x)
# X_v2 = v2_proj.fit_transform(x)


# Get random view splits
#...

# Do anomaly detection

