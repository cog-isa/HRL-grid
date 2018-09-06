# file = open('../experiments/plain DQN/experiment dataset2/obs_dataset.txt', "r")
# data = file.read()
# # print(data.split("next")[0])

import numpy as np

d = np.load('../experiments/plain DQN/experiment task1 dataset/obs_dataset.npy')

print(d[0].shape)

y_datatset = np.load('../experiments/plain DQN/experiment task1 dataset/done_dataset.npy')

print(np.sum(y_datatset))

# print(np.unique(d, axis = 0).shape)
print((d[0]==d[1]).all())

print(np.array_equal(d[0], d[2]))

d2 = np.load('../experiments/plain DQN/experiment task2 dataset/obs_dataset.npy')

print(d2.shape)