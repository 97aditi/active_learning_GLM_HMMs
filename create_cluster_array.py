import numpy as np

chain_lengths = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
seeds = [0, 1, 2, 3, 4]


cluster_array = []
for i, chain_length in enumerate(chain_lengths):
    for j, seed in enumerate(seeds):
        cluster_array.append([chain_length, seed])


print(len(cluster_array))
#save as npy
np.save('cluster_array.npy', cluster_array)