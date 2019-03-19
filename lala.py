import numpy as np

total = 70
frac = 0.9

batch_sizes = np.random.binomial(total, p=frac,size=(100,))


print(batch_sizes)