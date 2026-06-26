import numpy as np
d = np.load('data_ai/dataset.npz')
print("displacements:", d['displacements'].shape)
print("coefficients:", d['coefficients'].shape)
