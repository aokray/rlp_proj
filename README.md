# Random Linear Projection Project
Usage:

```
# If importing:
# from rlp import RLP

# k = reduced num of features
k = 2

# Random mode is the method for creating the random projection matrix
rand_mode = 'gaussian'

# This dict defines possible parameter values, default for now
d = {'mean'= 0, 'sd' = 1}

# X is some numpy dataset with shape n x p, for n instances and p features
x = ...

rand_projector = RLP(2, rand_mode)
transformed_data = rand_projector.fit_transform(x, **d)

# Transformed data has shape n x k
```